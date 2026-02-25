import torch
from .omni_mlp import TritonSwiGLUFunc
from .omni_router import OmniRouterFunc
from .omni_expert import OmniExpertFunc


def triton_omni_mlp_func(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Omni MLP function using Triton kernels.

    :param x: Input tensor of shape (num_tokens, hidden_size)
    :param gate_weight: Gate weight tensor of shape (intermediate_size, hidden_size)
    :param up_weight: Up weight tensor of shape (intermediate_size, hidden_size)
    :param down_weight: Down weight tensor of shape (hidden_size, intermediate_size)

    :return y: Output tensor of shape (num_tokens, hidden_size)
    """
    return torch.matmul(
        TritonSwiGLUFunc.apply(
            torch.matmul(x, gate_weight.t()),
            torch.matmul(x, up_weight.t()),
        ),
        down_weight.t(),
    )


def triton_omni_router_func(
    router_logits_x: torch.Tensor,
    router_logits_y: torch.Tensor,
    num_expert_sqrt: int,
    num_experts_per_token: int,
):
    """
    Omni router function using triton kernels.

    :param router_logits_x: Router logits for row dimension of shape (num_tokens, num_expert_sqrt)
    :param router_logits_y: Router logits for column dimension of shape (num_tokens, num_expert_sqrt)
    :param num_expert_sqrt: The square root of the number of experts
    :param num_experts_per_token: The number of experts assigned to each token

    :return scores: The routing scores for each token and expert of shape (num_tokens, num_experts_per_token)
    :return indices: The indices of the selected experts for each token of shape (num_tokens, num_experts_per_token)
    """
    return OmniRouterFunc.apply(
        router_logits_x, router_logits_y, num_expert_sqrt, num_experts_per_token
    )


def triton_omni_expert_func(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    routing_weight: torch.Tensor,
    indices: torch.Tensor,
):
    """
    Omni expert function using triton kernels.

    :param x: Input tensor of shape (num_tokens, hidden_size)
    :param up_weight: Up weight tensor of shape (num_experts, hidden_size)
    :param down_weight: Down weight tensor of shape (num_experts, hidden_size)
    :param routing_weight: Routing weight tensor of shape (num_tokens, num_experts_per_token)
    :param indices: Indices of the selected experts for each token of shape (num_tokens, num_experts_per_token)

    :return y: Output tensor of shape (num_tokens, hidden_size)
    """
    return OmniExpertFunc.apply(x, up_weight, down_weight, routing_weight, indices)
