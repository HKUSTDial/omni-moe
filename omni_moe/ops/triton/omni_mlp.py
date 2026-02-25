import torch
import triton
import triton.language as tl

from . import utils, activations


@triton.autotune(
    configs=utils.get_mlp_fwd_autotune_configs(),
    key=utils.MLP_FWD_AUTOTUNE_KEYS,
)
@triton.jit
def _fwd_kernel(
    A,
    B,
    C,
    stride_a,
    stride_b,
    stride_c,
    intermediate_size: tl.constexpr,
    TILE_K: tl.constexpr,
):
    hidden_idx = tl.program_id(0).to(tl.int64)

    # Initialize offsets
    offs_k = tl.arange(0, TILE_K).to(tl.int64)
    mask = offs_k < intermediate_size

    # Initialize pointers
    a_ptrs = A + hidden_idx * stride_a + offs_k
    b_ptrs = B + hidden_idx * stride_b + offs_k
    c_ptrs = C + hidden_idx * stride_c + offs_k

    # Load a and b
    a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    # Compute c
    c = activations.silu(a).to(b.dtype) * b

    # Store c
    tl.store(c_ptrs, c, mask=mask)


@triton.autotune(
    configs=utils.get_mlp_bwd_autotune_configs(),
    key=utils.MLP_BWD_AUTOTUNE_KEYS,
)
@triton.jit
def _bwd_kernel(
    dC,
    A,
    B,
    dA,
    dB,
    stride_dc,
    stride_a,
    stride_b,
    stride_da,
    stride_db,
    intermediate_size: tl.constexpr,
    TILE_K: tl.constexpr,
):
    hidden_idx = tl.program_id(0).to(tl.int64)

    # Initialize offsets
    offs_k = tl.arange(0, TILE_K).to(tl.int64)
    mask = offs_k < intermediate_size

    # Initialize pointers
    dc_ptrs = dC + hidden_idx * stride_dc + offs_k
    a_ptrs = A + hidden_idx * stride_a + offs_k
    b_ptrs = B + hidden_idx * stride_b + offs_k
    da_ptrs = dA + hidden_idx * stride_da + offs_k
    db_ptrs = dB + hidden_idx * stride_db + offs_k

    # Load dc, a, b
    dc = tl.load(dc_ptrs, mask=mask, other=0.0)
    a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    # recomputation to save memory
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a

    # Compute da and db
    db = dc * silu_a
    da = dc * (silu_a * (1 - sig_a) + sig_a) * b

    # Store da and db
    tl.store(da_ptrs, da, mask=mask)
    tl.store(db_ptrs, db, mask=mask)


def omni_swiglu_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
):
    # Assert inputs
    utils.assert_omni_mlp_fwd_inputs(gate, up)

    num_tokens, intermediate_size = gate.shape

    # Allocate output
    out = torch.empty_like(gate)

    grid = (num_tokens,)

    _fwd_kernel[grid](
        gate,
        up,
        out,
        gate.stride(-2),
        up.stride(-2),
        out.stride(-2),
        intermediate_size=intermediate_size,
    )

    return gate, up, out


def omni_swiglu_backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    do: torch.Tensor,
):
    num_tokens, intermediate_size = gate.shape

    # Allocate gradients
    dg = torch.empty_like(gate)
    du = torch.empty_like(up)

    grid = (num_tokens,)

    _bwd_kernel[grid](
        do,
        gate,
        up,
        dg,
        du,
        do.stride(-2),
        gate.stride(-2),
        up.stride(-2),
        dg.stride(-2),
        du.stride(-2),
        intermediate_size=intermediate_size,
    )

    return dg, du


class TritonSwiGLUFunc(torch.autograd.Function):
    @staticmethod
    @utils.ensure_contiguous
    def forward(ctx, gate, up):
        gate, up, o = omni_swiglu_forward(gate, up)
        ctx.save_for_backward(gate, up)
        return o

    @staticmethod
    @utils.ensure_contiguous
    def backward(ctx, do):
        gate, up = ctx.saved_tensors
        dg, du = omni_swiglu_backward(gate, up, do)
        return dg, du
