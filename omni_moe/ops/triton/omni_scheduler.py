from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SchedulingInfo:
    """
    Scheduling information for omni expert kernels.

    Attributes:
        tail_token_ids (torch.Tensor): token ids for tail pairs with shape (total_tail_pairs,)
        tail_expert_ids (torch.Tensor): expert ids for tail pairs with shape (num_unique_tail_experts,)
        tail_routing_weights (torch.Tensor): routing weights for tail pairs with shape (total_tail_pairs,)
        tail_offsets (torch.Tensor): pair segment offsets for tail experts with shape (num_unique_tail_experts + 1,)
        tail_sorted_pair_ids (torch.Tensor): sorted indices into original tail pairs with shape (total_tail_pairs,)
        num_tail_experts (int): number of unique tail experts
        max_tail_pairs_per_expert (int): maximum number of tail pairs per expert
        group_token_ids (torch.Tensor): token ids for grouped pairs with shape (total_group_tokens,)
        group_expert_ids (torch.Tensor): expert ids for grouped pairs with shape (num_groups, group_size)
        group_routing_weights (torch.Tensor): routing weights for grouped pairs with shape (total_group_tokens, group_size)
        group_offsets (torch.Tensor): token segment offsets for groups with shape (num_groups + 1,)
        group_sorted_pair_ids (torch.Tensor): sorted indices into original grouped pairs with shape (total_group_tokens, group_size)
        num_groups (int): number of groups
        max_group_tokens (int): maximum number of tokens in any group
    """

    # Tail info
    tail_token_ids: torch.Tensor = None
    tail_expert_ids: torch.Tensor = None
    tail_routing_weights: torch.Tensor = None
    tail_offsets: torch.Tensor = None
    tail_sorted_pair_ids: torch.Tensor = None
    num_tail_experts: int = 0
    max_tail_pairs_per_expert: int = 0

    # Group info
    group_token_ids: torch.Tensor = None
    group_expert_ids: torch.Tensor = None
    group_routing_weights: torch.Tensor = None
    group_offsets: torch.Tensor = None
    group_sorted_pair_ids: torch.Tensor = None
    num_groups: int = 0
    max_group_tokens: int = 0


# TODO: Triton has difficulty fusing these calculations into a kernel.
# For now, we are using Torch to implement the scheduling logic, but the computational efficiency of this should be improved in the future.
def get_scheduling_info(
    G: torch.Tensor,
    Indices: torch.Tensor,
    num_experts: int,
    group_size: int,
):
    device = G.device
    num_tokens, num_experts_per_token = G.shape

    # Determine if grouping is possible
    is_group = (num_tokens * num_experts_per_token / num_experts) > group_size**2

    # Initialize basic info
    token_ids = (
        torch.arange(
            num_tokens * num_experts_per_token, device=Indices.device, dtype=torch.int32
        )
        // num_experts_per_token
    )
    expert_ids = Indices.flatten().to(torch.int32)
    G = G.flatten()

    # Sort by expert ids
    sorted_expert_ids, sorted_pair_ids = torch.sort(expert_ids)
    sorted_pair_ids = sorted_pair_ids.to(torch.int32)

    # Get tail info for all pairs (may be split into groups below)
    tail_token_ids = token_ids[sorted_pair_ids]
    tail_routing_weights = G[sorted_pair_ids]

    # Build compressed expert list and offsets from sorted pairs
    tail_expert_ids, expert_counts = torch.unique_consecutive(
        sorted_expert_ids, return_counts=True
    )
    num_tail_experts = tail_expert_ids.numel()

    tail_offsets = torch.zeros(
        num_tail_experts + 1, dtype=torch.int32, device=Indices.device
    )
    tail_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    # Get max tail pairs per expert
    max_tail_pairs_per_expert = torch.max(expert_counts).item()

    if not is_group:
        return SchedulingInfo(
            tail_token_ids=tail_token_ids,
            tail_expert_ids=tail_expert_ids,
            tail_routing_weights=tail_routing_weights,
            tail_offsets=tail_offsets,
            tail_sorted_pair_ids=sorted_pair_ids,
            num_tail_experts=num_tail_experts,
            max_tail_pairs_per_expert=max_tail_pairs_per_expert,
        )

    # Build padded token matrix for clustering
    tokens_matrix = torch.full(
        (num_tail_experts, max_tail_pairs_per_expert),
        -1,
        dtype=tail_token_ids.dtype,
        device=device,
    )

    expert_row_ids = torch.repeat_interleave(
        torch.arange(num_tail_experts, device=device), expert_counts
    )
    start_offsets = torch.repeat_interleave(tail_offsets[:-1], expert_counts)
    positions = torch.arange(tail_token_ids.numel(), device=device) - start_offsets
    tokens_matrix[expert_row_ids, positions] = tail_token_ids

    # Cluster experts by identical token lists
    _, inverse = torch.unique(tokens_matrix, dim=0, return_inverse=True)
    sort_idx = torch.argsort(inverse)
    sorted_cluster = inverse[sort_idx]
    _, counts = torch.unique_consecutive(sorted_cluster, return_counts=True)
    cluster_offsets = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=device)
    if counts.numel() > 0:
        cluster_offsets[1:] = torch.cumsum(counts, dim=0)
    cluster_starts = torch.repeat_interleave(cluster_offsets[:-1], counts)
    pos_in_cluster = torch.arange(num_tail_experts, device=device) - cluster_starts
    full_group_sizes = (counts // group_size) * group_size
    keep_mask = pos_in_cluster < torch.repeat_interleave(full_group_sizes, counts)
    kept_sorted_idx = sort_idx[keep_mask]
    num_groups = int(kept_sorted_idx.numel() // group_size)

    # Build group outputs
    if num_groups > 0:
        group_rows = kept_sorted_idx.reshape(num_groups, group_size)
        group_expert_ids = tail_expert_ids[group_rows]
        group_rep_rows = group_rows[:, 0]
        group_lengths = expert_counts[group_rep_rows]
        max_group_len = (
            int(group_lengths.max().item()) if group_lengths.numel() > 0 else 0
        )

        pos_range = torch.arange(max_group_len, device=device)
        token_mask = pos_range.unsqueeze(0) < group_lengths.unsqueeze(1)

        # Shared within group, use representative expert
        rep_idx = tail_offsets[group_rep_rows].unsqueeze(1) + pos_range.unsqueeze(0)
        group_token_ids = tail_token_ids[rep_idx[token_mask]]

        group_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device=device)
        group_offsets[1:] = torch.cumsum(group_lengths, dim=0)

        # Index all experts in each group via offsets
        all_starts = tail_offsets[group_rows]  # (num_groups, group_size)
        valid_idx = (all_starts.unsqueeze(1) + pos_range.reshape(1, -1, 1))[
            token_mask
        ]  # (total_group_tokens, group_size)
        group_sorted_pair_ids = sorted_pair_ids[valid_idx]
        group_routing_weights = tail_routing_weights[valid_idx]
        max_group_tokens = max_group_len
    else:
        group_expert_ids = torch.empty(
            (0, group_size), dtype=tail_expert_ids.dtype, device=device
        )
        group_token_ids = torch.empty((0,), dtype=tail_token_ids.dtype, device=device)
        group_offsets = torch.zeros((1,), dtype=torch.int32, device=device)
        group_sorted_pair_ids = torch.empty(
            (0, group_size), dtype=sorted_pair_ids.dtype, device=device
        )
        group_routing_weights = torch.empty(
            (0, group_size), dtype=tail_routing_weights.dtype, device=device
        )
        max_group_tokens = 0

    # Build tail outputs from remaining experts not included in groups
    group_used_mask = torch.zeros(num_tail_experts, dtype=torch.bool, device=device)
    if kept_sorted_idx.numel() > 0:
        group_used_mask[kept_sorted_idx] = True
    tail_row_mask = ~group_used_mask
    tail_expert_ids_final = tail_expert_ids[tail_row_mask]
    tail_lengths = expert_counts[tail_row_mask]
    num_tail_experts_final = tail_expert_ids_final.numel()
    if num_tail_experts_final > 0:
        tail_indices = torch.where(tail_row_mask)[0]
        tail_starts = tail_offsets[tail_indices]
        max_tail_len = int(tail_lengths.max().item())
        tail_pos = torch.arange(max_tail_len, device=device)
        tail_valid = tail_pos.unsqueeze(0) < tail_lengths.unsqueeze(1)
        flat_idx = (tail_starts.unsqueeze(1) + tail_pos.unsqueeze(0))[tail_valid]
        tail_token_ids_final = tail_token_ids[flat_idx]
        tail_sorted_pair_ids_final = sorted_pair_ids[flat_idx]
        tail_routing_weights_final = tail_routing_weights[flat_idx]
        tail_offsets_final = torch.zeros(
            num_tail_experts_final + 1, dtype=torch.int32, device=device
        )
        tail_offsets_final[1:] = torch.cumsum(tail_lengths.to(torch.int32), dim=0)
        max_tail_pairs_per_expert_final = max_tail_len
    else:
        tail_token_ids_final = torch.empty(
            (0,), dtype=tail_token_ids.dtype, device=device
        )
        tail_sorted_pair_ids_final = torch.empty(
            (0,), dtype=sorted_pair_ids.dtype, device=device
        )
        tail_routing_weights_final = torch.empty(
            (0,), dtype=tail_routing_weights.dtype, device=device
        )
        tail_offsets_final = torch.zeros((1,), dtype=torch.int32, device=device)
        max_tail_pairs_per_expert_final = 0

    return SchedulingInfo(
        tail_token_ids=tail_token_ids_final,
        tail_expert_ids=tail_expert_ids_final,
        tail_routing_weights=tail_routing_weights_final,
        tail_offsets=tail_offsets_final,
        tail_sorted_pair_ids=tail_sorted_pair_ids_final,
        num_tail_experts=num_tail_experts_final,
        max_tail_pairs_per_expert=max_tail_pairs_per_expert_final,
        group_token_ids=group_token_ids,
        group_expert_ids=group_expert_ids,
        group_routing_weights=group_routing_weights,
        group_offsets=group_offsets,
        group_sorted_pair_ids=group_sorted_pair_ids,
        num_groups=num_groups,
        max_group_tokens=max_group_tokens,
    )
