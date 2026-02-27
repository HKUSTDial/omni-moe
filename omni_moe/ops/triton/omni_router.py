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
    I_X,
    I_Y,
    S,
    INDICES,
    stride_sxm,
    stride_sxn,
    stride_sym,
    stride_syn,
    stride_ixm,
    stride_ixn,
    stride_iym,
    stride_iyn,
    stride_sm,
    stride_sk,
    stride_im,
    stride_ik,
    num_tokens,
    topk: tl.constexpr,
    num_expert_sqrt: tl.constexpr,
    num_experts_per_token: tl.constexpr,
    topk_square: tl.constexpr,
    TILE_M: tl.constexpr,
):
    m_block = tl.program_id(0)

    # Initialize offsets
    offs_n = tl.arange(0, topk_square)

    # Compute expert coordinates
    ix = offs_n // topk
    iy = offs_n - ix * topk

    # TODO: Can't vectorize top-k in triton now, only slow loop can be used
    # I think top-k based router is difficult to implement efficiently in triton
    # we need to be improved to another better algorithm in the future
    # Loop-based topk O(k) iterations, each with max+argmax
    # This is faster than bitonic sort when k << n
    mask_n = offs_n < topk**2
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
        # For triton, loop is faster than unrolling when num_experts_per_token is small
        for k in range(num_experts_per_token):
            # Find max score and index
            max_score = tl.max(scores, axis=0)
            max_local_index = tl.argmax(scores, axis=0)

            # Remap local index to global index
            local_ix = max_local_index // topk
            local_iy = max_local_index - local_ix * topk
            indices_x = tl.load(
                I_X + m_idx * stride_ixm + local_ix * stride_ixn, mask=mask_m, other=0
            )
            indices_y = tl.load(
                I_Y + m_idx * stride_iym + local_iy * stride_iyn, mask=mask_m, other=0
            )
            max_index = indices_x * num_expert_sqrt + indices_y

            # Store score and index
            tl.store(scores_ptr + k * stride_sk, max_score, mask=mask_m)
            tl.store(indices_ptr + k * stride_ik, max_index, mask=mask_m)

            # Set selected scores to -inf for next iteration
            scores = tl.where(offs_n == max_local_index, -float("inf"), scores)


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
    topk = min(num_expert_sqrt, num_experts_per_token)
    topk_square = triton.next_power_of_2(topk**2)

    # For efficiency, we use torch.topk to get candidate experts
    # Because triton's topk does not return indices, and we need indices for the router
    topk_scores_x, topk_indices_x = torch.topk(router_logits_x, topk, dim=-1)
    topk_scores_y, topk_indices_y = torch.topk(router_logits_y, topk, dim=-1)
    topk_indices_x = topk_indices_x.to(torch.int32)
    topk_indices_y = topk_indices_y.to(torch.int32)

    # Allocate outputs
    scores = torch.empty(
        (num_tokens, num_experts_per_token),
        device=router_logits_x.device,
        dtype=router_logits_x.dtype,
    )
    indices = torch.empty(
        (num_tokens, num_experts_per_token),
        device=router_logits_x.device,
        dtype=torch.int32,
    )

    def grid(META):
        return (triton.cdiv(num_tokens, META["TILE_M"]),)

    _fwd_kernel[grid](
        topk_scores_x,
        topk_scores_y,
        topk_indices_x,
        topk_indices_y,
        scores,
        indices,
        topk_scores_x.stride(0),
        topk_scores_x.stride(1),
        topk_scores_y.stride(0),
        topk_scores_y.stride(1),
        topk_indices_x.stride(0),
        topk_indices_x.stride(1),
        topk_indices_y.stride(0),
        topk_indices_y.stride(1),
        scores.stride(0),
        scores.stride(1),
        indices.stride(0),
        indices.stride(1),
        num_tokens,
        topk,
        num_expert_sqrt,
        num_experts_per_token,
        topk_square,
    )

    return scores, indices


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
