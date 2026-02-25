import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from omni_moe.ops.triton import (
    triton_omni_mlp_func,
    triton_omni_router_func,
    triton_omni_expert_func,
)


class OmniMoEConfig:
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        num_experts: int,
        num_experts_per_token: int,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token


class OmniMoE(nn.Module):
    def __init__(self, config: OmniMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.num_experts = config.num_experts
        self.num_expert_sqrt = math.floor(math.sqrt(self.num_experts))
        self.top_k = config.num_experts_per_token

        # shared expert
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # router gate for retrieval experts
        self.router_gate_x = nn.Linear(
            self.hidden_size, self.num_expert_sqrt, bias=False
        )
        self.router_gate_y = nn.Linear(
            self.hidden_size, self.num_expert_sqrt, bias=False
        )
        self.router_norm_x = nn.BatchNorm1d(self.num_expert_sqrt, affine=False)
        self.router_norm_y = nn.BatchNorm1d(self.num_expert_sqrt, affine=False)

        # routed experts
        self.up_embed = nn.Embedding(self.num_experts, self.hidden_size)
        self.down_embed = nn.Embedding(self.num_experts, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)

        # get routing logits with router gate
        router_logits_x = self.router_norm_x(self.router_gate_x(hidden_states))
        router_logits_y = self.router_norm_y(self.router_gate_y(hidden_states))

        # Calculate log probabilities for routing
        # We use log_softmax because for product keys
        # log(P(x, y)) = log(P(x)) + log(P(y))
        # This allows us to use the efficient addition structure while working with probabilities
        router_log_probs_x = F.log_softmax(router_logits_x, dim=-1)
        router_log_probs_y = F.log_softmax(router_logits_y, dim=-1)

        # get experts with the highest routing probabilities
        scores, indices = triton_omni_router_func(
            router_log_probs_x, router_log_probs_y, self.num_expert_sqrt, self.top_k
        )

        # Convert log-probabilities back to probabilities
        routing_weights = torch.exp(scores)

        # mix routed experts states with shared expert states
        experts_states = triton_omni_expert_func(
            hidden_states,
            self.up_embed.weight,
            self.down_embed.weight,
            routing_weights,
            indices,
        )
        hidden_states = triton_omni_mlp_func(
            hidden_states,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
        )
        hidden_states = hidden_states + experts_states

        hidden_states = hidden_states.view(bsz, seq_len, -1)

        return hidden_states


class OmniMoE_Torch(nn.Module):
    def __init__(self, config: OmniMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.num_experts = config.num_experts
        self.num_expert_sqrt = math.floor(math.sqrt(self.num_experts))
        self.top_k = config.num_experts_per_token

        # shared expert
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # router gate for retrieval experts
        self.router_gate_x = nn.Linear(
            self.hidden_size, self.num_expert_sqrt, bias=False
        )
        self.router_gate_y = nn.Linear(
            self.hidden_size, self.num_expert_sqrt, bias=False
        )
        self.router_norm_x = nn.BatchNorm1d(self.num_expert_sqrt, affine=False)
        self.router_norm_y = nn.BatchNorm1d(self.num_expert_sqrt, affine=False)

        # routed experts
        self.up_embed = nn.Embedding(self.num_experts, self.hidden_size)
        self.down_embed = nn.Embedding(self.num_experts, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        # get routing logits with router gate
        router_logits_x = self.router_norm_x(
            self.router_gate_x(hidden_states.view(-1, self.hidden_size))
        )
        router_logits_y = self.router_norm_y(
            self.router_gate_y(hidden_states.view(-1, self.hidden_size))
        )

        # Calculate log probabilities for routing
        # We use log_softmax because for Product Keys, P(x, y) = P(x) * P(y)
        # log(P(x, y)) = log(P(x)) + log(P(y))
        # This allows us to use the efficient addition structure while working with probabilities
        router_log_probs_x = F.log_softmax(router_logits_x, dim=-1)
        router_log_probs_y = F.log_softmax(router_logits_y, dim=-1)

        # get experts with the highest routing probabilities
        scores_x, indices_x = router_log_probs_x.topk(self.num_expert_sqrt, dim=-1)
        scores_y, indices_y = router_log_probs_y.topk(self.num_expert_sqrt, dim=-1)
        all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
        all_indices = indices_x.unsqueeze(
            -1
        ) * self.num_expert_sqrt + indices_y.unsqueeze(-2)
        all_scores = all_scores.view(*all_scores.shape[:-2], -1)
        all_indices = all_indices.view(*all_indices.shape[:-2], -1)
        scores, position_indices = all_scores.topk(self.top_k, dim=-1)
        indices = all_indices.gather(-1, position_indices)

        # Convert log-probabilities back to probabilities
        routing_weights = torch.exp(scores)

        # mix routed experts states with shared expert states
        up_embed = self.up_embed(indices)
        down_embed = self.down_embed(indices)
        experts_weights = torch.matmul(
            up_embed, hidden_states.view(bsz * seq_len, -1, 1)
        ).view(bsz * seq_len, -1)
        experts_weights = self.act_fn(experts_weights) * routing_weights
        experts_states = torch.matmul(
            experts_weights.view(bsz * seq_len, 1, -1), down_embed
        ).view(bsz, seq_len, -1)
        hidden_states = self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        hidden_states = hidden_states + experts_states
        return hidden_states
