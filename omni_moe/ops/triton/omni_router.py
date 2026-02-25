import torch
import triton
from triton import language as tl

from . import utils


@triton.autotune(
    configs=utils.get_router_fwd_autotune_configs(),
    key=utils.ROUTER_FWD_AUTOTUNE_KEYS,
)
@triton.jit
def _fwd_kernel(
    S_X,
    S_Y,
    S,
    INDICES,
    stride_sxm,
    stride_sxn,
    stride_sym,
    stride_syn,
    stride_sm,
    stride_sk,
    stride_im,
    stride_ik,
    num_tokens,
    num_expert_sqrt: tl.constexpr,
    num_experts_per_token: tl.constexpr,
    num_experts: tl.constexpr,
    TILE_M: tl.constexpr,
):
    m_block = tl.program_id(0)

    # Initialize offsets
    offs_n = tl.arange(0, num_experts)

    # Compute expert coordinates
    ix = offs_n // num_expert_sqrt
    iy = offs_n - ix * num_expert_sqrt

    # TODO: Can't vectorize top-k in triton now, only slow loop can be used
    # I think top-k based router is difficult to implement efficiently in triton
    # we need to be improved to another better algorithm in the future
    # Loop-based topk O(k) iterations, each with max+argmax
    # This is faster than bitonic sort when k << n
    mask_n = offs_n < num_experts
    for m in range(TILE_M):
        m_idx = m_block * TILE_M + m
        mask_m = m_idx < num_tokens
        mask = mask_n & mask_m

        # Initialize pointers
        scores_x_ptrs = S_X + m_idx * stride_sxm + ix * stride_sxn
        scores_y_ptrs = S_Y + m_idx * stride_sym + iy * stride_syn
        scores_ptr = S + m_idx * stride_sm
        indices_ptr = INDICES + m_idx * stride_im

        # Load scores_x and scores_y
        scores_x = tl.load(scores_x_ptrs, mask=mask, other=-float("inf"))
        scores_y = tl.load(scores_y_ptrs, mask=mask, other=-float("inf"))

        # Compute combined scores
        scores = scores_x + scores_y

        # Top-k selection
        # For triton, loop is faster than unrolling when num_experts is small
        for k in range(num_experts_per_token):
            # Find max score and index
            max_score = tl.max(scores, axis=0)
            max_index = tl.argmax(scores, axis=0)

            # Store score and index
            tl.store(scores_ptr + k * stride_sk, max_score, mask=mask_m)
            tl.store(indices_ptr + k * stride_ik, max_index, mask=mask_m)

            # Set selected scores to -inf for next iteration
            scores = tl.where(offs_n == max_index, -float("inf"), scores)


@triton.autotune(
    configs=utils.get_router_fwd_split_experts_autotune_configs(),
    key=utils.ROUTER_FWD_SPLIT_EXPERTS_AUTOTUNE_KEYS,
)
@triton.jit
def _fwd_split_experts_kernel(
    S_X,
    S_Y,
    S,
    INDICES,
    stride_sxm,
    stride_sxn,
    stride_sym,
    stride_syn,
    stride_sm,
    stride_sk,
    stride_im,
    stride_ik,
    num_tokens,
    num_expert_sqrt: tl.constexpr,
    num_experts_per_token: tl.constexpr,
    num_experts: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    m_block = tl.program_id(0)

    # Initialize offsets
    offs_nb = tl.arange(0, TILE_N)
    offs_k = tl.arange(0, num_experts_per_token)

    # Loop over tokens in blocks of TILE_M
    # TODO: same issue of top-k as above, need better algorithm
    for m in range(TILE_M):
        m_idx = m_block * TILE_M + m
        mask_m = m_idx < num_tokens

        # Initialize pointers
        scores_x_ptr = S_X + m_idx * stride_sxm
        scores_y_ptr = S_Y + m_idx * stride_sym
        scores_ptr = S + m_idx * stride_sm + offs_k * stride_sk
        indices_ptr = INDICES + m_idx * stride_im + offs_k * stride_ik

        # Initialize scores and indices
        topk_scores = tl.full((num_experts_per_token,), -float("inf"), dtype=tl.float32)
        topk_indices = tl.full((num_experts_per_token,), -1, dtype=tl.int32)

        # Loop over experts in blocks of TILE_N
        for start_expert in range(0, num_experts, TILE_N):
            start_expert = tl.multiple_of(start_expert, TILE_N)
            offs_n = offs_nb + start_expert

            # Compute expert coordinates
            ix = offs_n // num_expert_sqrt
            iy = offs_n - ix * num_expert_sqrt

            # Create mask
            mask_n = offs_n < num_experts
            mask = mask_n & mask_m

            # Load scores_x and scores_y
            scores_x = tl.load(
                scores_x_ptr + ix * stride_sxn,
                mask=mask,
                other=-float("inf"),
            )
            scores_y = tl.load(
                scores_y_ptr + iy * stride_syn,
                mask=mask,
                other=-float("inf"),
            )

            # Compute combined scores
            scores = (scores_x + scores_y).to(tl.float32)

            # Top-k selection within this block
            for k in range(num_experts_per_token):
                # Find max score and index
                max_score = tl.max(scores, axis=0)
                max_index = tl.argmax(scores, axis=0)

                # Get current top-k score for comparison
                current_score = tl.where(offs_k == k, topk_scores, -float("inf"))
                current_max_score = tl.max(current_score, axis=0)

                # Update top-k if found a better score
                if max_score > current_max_score:
                    # Shift down lower scores in top-k
                    for shift_k in range(num_experts_per_token - 1, k, -1):
                        prev_score = tl.where(offs_k == shift_k - 1, topk_scores, 0.0)
                        prev_index = tl.where(offs_k == shift_k - 1, topk_indices, -1)
                        topk_scores = tl.where(
                            offs_k == shift_k,
                            tl.max(prev_score, axis=0),
                            topk_scores,
                        )
                        topk_indices = tl.where(
                            offs_k == shift_k,
                            tl.max(prev_index, axis=0),
                            topk_indices,
                        )
                    topk_scores = tl.where(offs_k == k, max_score, topk_scores)
                    topk_indices = tl.where(
                        offs_k == k, max_index + start_expert, topk_indices
                    )

                    # Mask out selected element from scores
                    scores = tl.where(offs_nb == max_index, -float("inf"), scores)

        # Store final top-k scores and indices
        mask = mask_m & (offs_k < num_experts_per_token)
        tl.store(scores_ptr, topk_scores, mask=mask)
        tl.store(indices_ptr, topk_indices, mask=mask)


@triton.autotune(
    configs=utils.get_router_bwd_autotune_configs(),
    key=utils.ROUTER_BWD_AUTOTUNE_KEYS,
    reset_to_zero=["DS_X", "DS_Y"],
)
@triton.jit
def _bwd_kernel(
    DS,
    INDICES,
    DS_X,
    DS_Y,
    stride_dsm,
    stride_dsk,
    stride_im,
    stride_ik,
    stride_dsxm,
    stride_dsxn,
    stride_dsym,
    stride_dsyn,
    num_tokens,
    num_expert_sqrt: tl.constexpr,
    num_experts_per_token: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    m_block = tl.program_id(0)
    k_block = tl.program_id(1)

    # Initialize offsets
    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
    offs_k = k_block * TILE_K + tl.arange(0, TILE_K)

    # Initialize pointers
    dscores_ptr = DS + offs_m[:, None] * stride_dsm + offs_k[None, :] * stride_dsk
    indices_ptr = INDICES + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    dscores_x_ptr = DS_X + offs_m[:, None] * stride_dsxm
    dscores_y_ptr = DS_Y + offs_m[:, None] * stride_dsym

    # Create masks
    mask_m = offs_m < num_tokens
    mask_k = offs_k < num_experts_per_token
    mask = mask_m[:, None] & mask_k[None, :]

    # Load dscores
    dscores = tl.load(dscores_ptr, mask=mask, other=0.0)

    # Load indices
    indices = tl.load(indices_ptr, mask=mask, other=-1)

    # Convert to float32 for atomic accumulation
    dscores = dscores.to(tl.float32)

    # Compute expert coordinates
    ix = indices // num_expert_sqrt
    iy = indices - ix * num_expert_sqrt

    # Atomic accumulation to dscores_x and dscores_y
    tl.atomic_add(
        dscores_x_ptr + ix * stride_dsxn,
        dscores,
        mask=mask,
    )
    tl.atomic_add(
        dscores_y_ptr + iy * stride_dsyn,
        dscores,
        mask=mask,
    )


def _omni_router_forward(
    router_logits_x: torch.Tensor,
    router_logits_y: torch.Tensor,
    num_expert_sqrt: int,
    num_experts_per_token: int,
):
    # Assert inputs
    utils.assert_omni_router_fwd_inputs(
        router_logits_x,
        router_logits_y,
        num_expert_sqrt,
        num_experts_per_token,
    )
    num_tokens = router_logits_x.size(0)

    # Allocate outputs
    scores = torch.empty(
        (num_tokens, num_experts_per_token),
        device=router_logits_x.device,
        dtype=torch.float32,
    )
    indices = torch.empty(
        (num_tokens, num_experts_per_token),
        device=router_logits_x.device,
        dtype=torch.int32,
    )

    # Compute total number of experts
    num_experts = triton.next_power_of_2(num_expert_sqrt * num_expert_sqrt)

    def grid(META):
        return (triton.cdiv(num_tokens, META["TILE_M"]),)

    # If num_expert_sqrt is small, use lightweight kernel
    if num_expert_sqrt < 128:
        _fwd_kernel[grid](
            router_logits_x,
            router_logits_y,
            scores,
            indices,
            router_logits_x.stride(0),
            router_logits_x.stride(1),
            router_logits_y.stride(0),
            router_logits_y.stride(1),
            scores.stride(0),
            scores.stride(1),
            indices.stride(0),
            indices.stride(1),
            num_tokens,
            num_expert_sqrt,
            num_experts_per_token,
            num_experts,
        )
    else:
        _fwd_split_experts_kernel[grid](
            router_logits_x,
            router_logits_y,
            scores,
            indices,
            router_logits_x.stride(0),
            router_logits_x.stride(1),
            router_logits_y.stride(0),
            router_logits_y.stride(1),
            scores.stride(0),
            scores.stride(1),
            indices.stride(0),
            indices.stride(1),
            num_tokens,
            num_expert_sqrt,
            num_experts_per_token,
            num_experts,
        )

    return scores.to(router_logits_x.dtype), indices


def _omni_router_backward(
    dscores: torch.Tensor,
    indices: torch.Tensor,
    num_expert_sqrt: int,
):
    num_tokens, num_experts_per_token = dscores.shape

    # We use float32 for accumulation to reduce numerical issues
    dscores_x = torch.empty(
        (num_tokens, num_expert_sqrt), device=dscores.device, dtype=torch.float32
    )
    dscores_y = torch.empty(
        (num_tokens, num_expert_sqrt), device=dscores.device, dtype=torch.float32
    )

    def grid(META):
        return (
            triton.cdiv(num_tokens, META["TILE_M"]),
            triton.cdiv(num_experts_per_token, META["TILE_K"]),
        )

    _bwd_kernel[grid](
        dscores,
        indices,
        dscores_x,
        dscores_y,
        dscores.stride(0),
        dscores.stride(1),
        indices.stride(0),
        indices.stride(1),
        dscores_x.stride(0),
        dscores_x.stride(1),
        dscores_y.stride(0),
        dscores_y.stride(1),
        num_tokens,
        num_expert_sqrt,
        num_experts_per_token,
    )

    return dscores_x.to(dscores.dtype), dscores_y.to(dscores.dtype)


class OmniRouterFunc(torch.autograd.Function):
    @staticmethod
    @utils.ensure_contiguous
    def forward(
        ctx, router_logits_x, router_logits_y, num_expert_sqrt, num_experts_per_token
    ):
        scores, indices = _omni_router_forward(
            router_logits_x, router_logits_y, num_expert_sqrt, num_experts_per_token
        )
        ctx.save_for_backward(indices)
        ctx.num_expert_sqrt = num_expert_sqrt

        return scores, indices

    @staticmethod
    @utils.ensure_contiguous
    def backward(ctx, dscores, dindices):
        (indices,) = ctx.saved_tensors

        drouter_logits_x, drouter_logits_y = _omni_router_backward(
            dscores, indices, ctx.num_expert_sqrt
        )

        return drouter_logits_x, drouter_logits_y, None, None
