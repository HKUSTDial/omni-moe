import pytest
import torch

import testing
from omni_moe.ops.triton import triton_omni_mlp_func


def pytorch_mlp_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    orig_dtype = x.dtype
    gate_activations = torch.matmul(x, gate_weight.t())
    up_activations = torch.matmul(x, up_weight.t())
    hidden_states = (
        torch.nn.functional.silu(gate_activations.float()) * up_activations.float()
    ).to(orig_dtype)
    out = torch.matmul(hidden_states, down_weight.t())
    return out


def triton_mlp_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
):
    return triton_omni_mlp_func(x, gate_weight, up_weight, down_weight)


def pytorch_mlp_backward(
    loss: torch.Tensor,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
):
    loss.backward()
    return (
        x.grad,
        gate_weight.grad,
        up_weight.grad,
        down_weight.grad,
    )


def triton_mlp_backward(
    loss: torch.Tensor,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
):
    loss.backward()
    return (
        x.grad,
        gate_weight.grad,
        up_weight.grad,
        down_weight.grad,
    )


def make_forward_factory(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(42)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.5)
    gate_weight = torch.randn(
        intermediate_size, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    up_weight = torch.randn(
        intermediate_size, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    down_weight = torch.randn(
        hidden_size, intermediate_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)

    def _factory(_impl: testing.Implementation):
        args = (
            hidden_states.clone(),
            gate_weight.clone(),
            up_weight.clone(),
            down_weight.clone(),
        )
        kwargs = {}
        return args, kwargs

    return _factory


def make_backward_factory(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(42)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.5)
    gate_weight = torch.randn(
        intermediate_size, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    up_weight = torch.randn(
        intermediate_size, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    down_weight = torch.randn(
        hidden_size, intermediate_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)

    def _factory(impl: testing.Implementation):
        x = hidden_states.clone().detach().requires_grad_(True)
        gate = gate_weight.clone().detach().requires_grad_(True)
        up = up_weight.clone().detach().requires_grad_(True)
        down = down_weight.clone().detach().requires_grad_(True)

        if impl.backend == testing.Backend.PYTORCH:
            loss = pytorch_mlp_forward(x, gate, up, down).sum()
        elif impl.backend == testing.Backend.TRITON:
            loss = triton_mlp_forward(x, gate, up, down).sum()
        else:
            raise ValueError(f"Unknown backend: {impl.backend}")

        return (loss, x, gate, up, down), {}

    return _factory


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        # (num_tokens, hidden_size, intermediate_size)
        (1024, 1024, 1024),
        (1024, 1024, 4096),
        (1024, 1024, 16384),
        (1024, 1024, 65536),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mlp_forward_throughput(dtype: torch.dtype, case: tuple[int, int, int]) -> None:
    num_tokens, hidden_size, intermediate_size = case
    device = torch.device("cuda")

    print(
        f"[mlp forward] num_tokens={num_tokens}, hidden_size={hidden_size}, intermediate_size={intermediate_size}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_mlp_forward,
        triton_impl=triton_mlp_forward,
    )
    flops = 6.0 * num_tokens * hidden_size * intermediate_size
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    results = testing.run_benchmarks(
        impls,
        make_forward_factory(num_tokens, hidden_size, intermediate_size, device, dtype),
        flops=flops,
        config=config,
    )

    testing.show_benchmarks(results)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32],
)
@pytest.mark.parametrize(
    "case",
    [
        # (num_tokens, hidden_size, intermediate_size)
        (1024, 1024, 1024),
        (1024, 1024, 4096),
        (1024, 1024, 16384),
        (1024, 1024, 65536),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mlp_backward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int]
) -> None:
    num_tokens, hidden_size, intermediate_size = case
    device = torch.device("cuda")

    print(
        f"[mlp backward] num_tokens={num_tokens}, hidden_size={hidden_size}, intermediate_size={intermediate_size}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_mlp_backward,
        triton_impl=triton_mlp_backward,
    )
    flops = 2 * 6.0 * num_tokens * hidden_size * intermediate_size
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    results = testing.run_benchmarks(
        impls,
        make_backward_factory(
            num_tokens, hidden_size, intermediate_size, device, dtype
        ),
        flops=flops,
        config=config,
    )

    testing.show_benchmarks(results)
