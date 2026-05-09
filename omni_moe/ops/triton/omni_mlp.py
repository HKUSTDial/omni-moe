from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.flexpoint import (
    float_to_flex,
    load_scale,
    update_scale,
)
from triton_kernels.target_info import is_hip, num_sms

from . import utils


@dataclass(frozen=True)
class FlexCtx:
    out_data: OutFlexData = OutFlexData()
    inp_data: InFlexData = InFlexData()
    saturate_inf: bool = False


@dataclass(frozen=True)
class PrecisionConfig:
    limit: float = None
    flex_ctx: FlexCtx = FlexCtx()


@triton.jit
def exp_ftz(x):
    if tl.target_info.is_cuda():
        log2_e: tl.constexpr = 1.4426950408889634
        x *= log2_e
        return tl.inline_asm_elementwise(
            "ex2.approx.ftz.f32 $0, $1;",
            "=r, r",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        return tl.exp(x)


@triton.jit
def fast_sigmoid(x):
    return 1.0 / (1.0 + exp_ftz(-x))


@triton.jit
def compute_silu_mul(gate, up, scale, limit):
    gate = gate.to(tl.float32) * scale
    up = up.to(tl.float32) * scale
    if limit is not None:
        gate = tl.clamp(gate, -limit, limit)
        up = tl.clamp(up, -limit, limit)
    s = gate * fast_sigmoid(gate)
    return s * up


@triton.jit
def thread_local_absmax(x, BLOCK_SIZE: tl.constexpr, NUM_THREADS: tl.constexpr):
    return tl.max(
        tl.reshape(
            tl.abs(x), [NUM_THREADS, BLOCK_SIZE // NUM_THREADS], can_reorder=True
        ),
        axis=1,
    )


def omni_mlp_fwd_repr(specialization):
    constants = specialization.constants
    return f"_omni_mlp_fwd_{constants['BLOCK_M']}x{constants['BLOCK_N']}"


def omni_mlp_fwd_launch_metadata(grid, kernel, args):
    M, N = args["M"], args["N"]
    A = args["A"]
    return {
        "name": f"{kernel.name} [M={M}, N={N}]",
        "bytes": 2 * M * N * A.element_size() + M * N * A.element_size(),
    }


def omni_mlp_bwd_repr(specialization):
    constants = specialization.constants
    return f"_omni_mlp_bwd_{constants['BLOCK_M']}x{constants['BLOCK_N']}"


def omni_mlp_bwd_launch_metadata(grid, kernel, args):
    M, N = args["M"], args["N"]
    A = args["A"]
    return {
        "name": f"{kernel.name} [M={M}, N={N}]",
        "bytes": 3 * M * N * A.element_size() + 2 * M * N * A.element_size(),
    }


@triton.jit(repr=omni_mlp_fwd_repr, launch_metadata=omni_mlp_fwd_launch_metadata)
def _fwd_kernel(
    Out,
    OutExpectedScale,
    OutActualScale,
    OutChecksumScale,
    A,
    B,
    AScale,
    M,
    N,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    stride_om,
    stride_on,
    limit: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    M_BLOCKS,
    N_BLOCKS,
    flexpoint_saturate_inf: tl.constexpr,
):
    local_max = tl.full([tl.extra.cuda.num_threads()], 0.0, tl.float32)
    a_scale = load_scale(AScale)
    out_expected_scale = load_scale(OutExpectedScale)

    for pid in tl.range(
        tl.program_id(0), M_BLOCKS * N_BLOCKS, tl.num_programs(0), num_stages=2
    ):
        pid_m = pid // N_BLOCKS
        pid_n = pid % N_BLOCKS
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M

        if EVEN_N:
            mask = mask_m[:, None]
        else:
            mask = mask_m[:, None] & (offs_n[None, :] < N)

        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
            mask=mask,
            other=0.0,
        )
        b = tl.load(
            B + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn,
            mask=mask,
            other=0.0,
        )

        out = compute_silu_mul(a, b, a_scale, limit)

        if OutActualScale is not None:
            absmax = thread_local_absmax(out, out.numel, tl.extra.cuda.num_threads())
            local_max = tl.maximum(local_max, absmax)

        out = float_to_flex(
            out,
            out_expected_scale,
            None,
            OutChecksumScale,
            None,
            Out,
            flexpoint_saturate_inf,
        )
        tl.store(
            Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            out,
            mask=mask,
        )

    update_scale(local_max, OutActualScale, Out)


@triton.jit(repr=omni_mlp_bwd_repr, launch_metadata=omni_mlp_bwd_launch_metadata)
def _bwd_kernel(
    dC,
    A,
    B,
    dA,
    dB,
    M,
    N,
    stride_dcm,
    stride_dcn,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    stride_dam,
    stride_dan,
    stride_dbm,
    stride_dbn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    M_BLOCKS,
    N_BLOCKS,
):
    for pid in tl.range(
        tl.program_id(0), M_BLOCKS * N_BLOCKS, tl.num_programs(0), num_stages=2
    ):
        pid_m = pid // N_BLOCKS
        pid_n = pid % N_BLOCKS
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        if EVEN_N:
            mask = offs_m[:, None] < M
        else:
            mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        dc = tl.load(
            dC + offs_m[:, None] * stride_dcm + offs_n[None, :] * stride_dcn,
            mask=mask,
            other=0.0,
        )
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            B + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn,
            mask=mask,
            other=0.0,
        )

        sig_a = fast_sigmoid(a)
        silu_a = a * sig_a

        db = dc * silu_a
        da = dc * (silu_a * (1 - sig_a) + sig_a) * b

        tl.store(
            dA + offs_m[:, None] * stride_dam + offs_n[None, :] * stride_dan,
            da,
            mask=mask,
        )
        tl.store(
            dB + offs_m[:, None] * stride_dbm + offs_n[None, :] * stride_dbn,
            db,
            mask=mask,
        )


def omni_swiglu_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    precision_config: PrecisionConfig = PrecisionConfig(),
):
    utils.assert_omni_mlp_fwd_inputs(gate, up)
    M, N = gate.shape
    out = torch.empty_like(gate)

    flex_ctx = precision_config.flex_ctx
    BLOCK_M = 32 // gate.element_size()
    BLOCK_N = 128
    num_warps = 4
    kwargs = {"maxnreg": 64} if not is_hip() else {}

    M_BLOCKS = triton.cdiv(M, BLOCK_M)
    N_BLOCKS = triton.cdiv(N, BLOCK_N)
    n_sms = num_sms()

    if M_BLOCKS * N_BLOCKS >= 8 * n_sms:
        grid = (8 * n_sms,)
    else:
        grid = (min(M_BLOCKS * N_BLOCKS, 4 * n_sms),)

    _fwd_kernel[grid](
        flex_ctx.out_data.reinterpret(out),
        flex_ctx.out_data.expected_scale,
        flex_ctx.out_data.actual_scale,
        flex_ctx.out_data.checksum_scale,
        flex_ctx.inp_data.reinterpret(gate),
        flex_ctx.inp_data.reinterpret(up),
        flex_ctx.inp_data.scale,
        M,
        N,
        gate.stride(0),
        gate.stride(1),
        up.stride(0),
        up.stride(1),
        out.stride(0),
        out.stride(1),
        limit=precision_config.limit,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        EVEN_N=N % BLOCK_N == 0,
        M_BLOCKS=M_BLOCKS,
        N_BLOCKS=N_BLOCKS,
        flexpoint_saturate_inf=flex_ctx.saturate_inf,
        num_warps=num_warps,
        **kwargs,
    )

    return gate, up, out


def omni_swiglu_backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    do: torch.Tensor,
):
    M, N = gate.shape
    dg = torch.empty_like(gate)
    du = torch.empty_like(up)

    BLOCK_M = 32 // gate.element_size()
    BLOCK_N = 128
    num_warps = 4
    kwargs = {"maxnreg": 64} if not is_hip() else {}

    M_BLOCKS = triton.cdiv(M, BLOCK_M)
    N_BLOCKS = triton.cdiv(N, BLOCK_N)
    n_sms = num_sms()

    if M_BLOCKS * N_BLOCKS >= 8 * n_sms:
        grid = (8 * n_sms,)
    else:
        grid = (min(M_BLOCKS * N_BLOCKS, 4 * n_sms),)

    _bwd_kernel[grid](
        do,
        gate,
        up,
        dg,
        du,
        M,
        N,
        do.stride(0),
        do.stride(1),
        gate.stride(0),
        gate.stride(1),
        up.stride(0),
        up.stride(1),
        dg.stride(0),
        dg.stride(1),
        du.stride(0),
        du.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        EVEN_N=N % BLOCK_N == 0,
        M_BLOCKS=M_BLOCKS,
        N_BLOCKS=N_BLOCKS,
        num_warps=num_warps,
        **kwargs,
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
