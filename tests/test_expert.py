import pytest
import torch

import testing
from omni_moe.ops.triton import triton_omni_expert_func


def pytorch_expert_forward(
    hidden_states: torch.Tensor,
    up_embed: torch.Tensor,
    down_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
):
    num_tokens, _ = hidden_states.shape

    up_weights = up_embed[indices]
    down_weights = down_embed[indices]

    expert_weights = torch.matmul(up_weights, hidden_states.unsqueeze(-1)).view(
        num_tokens, -1
    )
    expert_weights = torch.nn.functional.silu(expert_weights) * routing_weights
    expert_states = torch.matmul(expert_weights.unsqueeze(-2), down_weights).squeeze(-2)
    return expert_states


def triton_expert_forward(
    hidden_states: torch.Tensor,
    up_embed: torch.Tensor,
    down_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
):
    expert_states = triton_omni_expert_func(
        hidden_states, up_embed, down_embed, routing_weights, indices
    )
    return expert_states


def pytorch_expert_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    up_embed: torch.Tensor,
    down_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
):
    loss.backward()
    return (
        hidden_states.grad,
        up_embed.grad,
        down_embed.grad,
        routing_weights.grad,
    )


def triton_expert_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    up_embed: torch.Tensor,
    down_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
):
    loss.backward()
    return (
        hidden_states.grad,
        up_embed.grad,
        down_embed.grad,
        routing_weights.grad,
    )


def make_forward_factory(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.5)
    up_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    down_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    routing_weights = torch.randn(
        num_tokens, top_k, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    indices = torch.randint(
        0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32
    )

    def _factory(_impl: testing.Implementation):
        args = (
            hidden_states.clone(),
            up_embed.clone(),
            down_embed.clone(),
            routing_weights.clone(),
            indices.clone(),
        )
        kwargs = {}
        return args, kwargs

    return _factory


def make_backward_factory(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.5)
    up_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    down_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    routing_weights = torch.randn(
        num_tokens, top_k, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    indices = torch.randint(
        0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32
    )

    def _factory(impl: testing.Implementation):
        hidden = hidden_states.clone().detach().requires_grad_(True)
        up = up_embed.clone().detach().requires_grad_(True)
        down = down_embed.clone().detach().requires_grad_(True)
        routing = routing_weights.clone().detach().requires_grad_(True)
        idx = indices.clone()

        if impl.backend == testing.Backend.PYTORCH:
            loss = pytorch_expert_forward(
                hidden,
                up,
                down,
                routing,
                idx,
            ).sum()
        elif impl.backend == testing.Backend.TRITON:
            loss = triton_expert_forward(
                hidden,
                up,
                down,
                routing,
                idx,
            ).sum()

        return (
            loss,
            hidden,
            up,
            down,
            routing,
            idx,
        ), {}

    return _factory


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        # (num_tokens, hidden_size, num_experts, top_k)
        (1024, 1024, 1024, 32),
        (1024, 1024, 4096, 32),
        (1024, 1024, 16384, 32),
        (1024, 1024, 65536, 32),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_expert_forward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int]
) -> None:
    num_tokens, hidden_size, num_experts, top_k = case
    device = torch.device("cuda")

    print(
        f"[expert forward] num_tokens={num_tokens}, hidden_size={hidden_size}, num_experts={num_experts}, top_k={top_k}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_expert_forward,
        triton_impl=triton_expert_forward,
    )
    flops = 4.0 * num_tokens * top_k * hidden_size
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    results = testing.run_benchmarks(
        impls,
        make_forward_factory(
            num_tokens,
            hidden_size,
            num_experts,
            top_k,
            device,
            dtype,
        ),
        flops=flops,
        config=config,
    )

    testing.show_benchmarks(results)


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        # (num_tokens, hidden_size, num_experts, top_k)
        (1024, 1024, 1024, 32),
        (1024, 1024, 4096, 32),
        (1024, 1024, 16384, 32),
        (1024, 1024, 65536, 32),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_expert_backward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int]
) -> None:
    num_tokens, hidden_size, num_experts, top_k = case
    device = torch.device("cuda")

    print(
        f"[expert backward] num_tokens={num_tokens}, hidden_size={hidden_size}, num_experts={num_experts}, top_k={top_k}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_expert_backward,
        triton_impl=triton_expert_backward,
    )
    flops = 2 * 4.0 * num_tokens * top_k * hidden_size
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    results = testing.run_benchmarks(
        impls,
        make_backward_factory(
            num_tokens,
            hidden_size,
            num_experts,
            top_k,
            device,
            dtype,
        ),
        flops=flops,
        config=config,
    )

    testing.show_benchmarks(results)
