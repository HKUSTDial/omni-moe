import functools
import torch
import triton


def get_device():
    """
    Get the appropriate device for computation.

    :return device: torch.device object
    """
    # TODO: add NPU
    # Works for both NVIDIA and AMD
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Intel XPU if available
    elif torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_arch(device: torch.device):
    """
    Get the architecture string for the given device.

    :param device: torch.device object

    :return arch: Architecture string
    """
    if device == torch.device("cuda"):
        capability = torch.cuda.get_device_capability(device)
        sm = f"sm{capability[0]}{capability[1]}"
        return f"cuda:{sm}"
    elif device == torch.device("xpu"):
        return "N/A"
    elif device == torch.device("mps"):
        return "N/A"
    elif device == torch.device("cpu"):
        return "N/A"
    else:
        raise ValueError(f"Unsupported device: {device}")


def ensure_contiguous(fn):
    """
    Decorator to ensure that all tensor inputs to the decorated function are contiguous.

    :param fn: Function to be decorated

    :return wrapper: Wrapped function
    """

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def assert_omni_mlp_fwd_inputs(
    gate: torch.Tensor,
    up: torch.Tensor,
):
    assert gate.dim() == 2, (
        "gate must be a 2D tensor of shape (batch_size * seq_len, hidden_size)"
    )
    assert up.dim() == 2, (
        "up must be a 2D tensor of shape (batch_size * seq_len, hidden_size)"
    )
    assert gate.shape == up.shape, "gate and up must have the same shape"


def assert_omni_router_fwd_inputs(
    router_logits_x: torch.Tensor,
    router_logits_y: torch.Tensor,
    num_expert_sqrt: int,
    num_experts_per_token: int,
):
    assert router_logits_x.dim() == 2, (
        "router_logits_x must be a 2D tensor of shape (batch_size * seq_len, num_expert_sqrt)"
    )
    assert router_logits_y.dim() == 2, (
        "router_logits_y must be a 2D tensor of shape (batch_size * seq_len, num_expert_sqrt)"
    )
    assert router_logits_x.size(0) == router_logits_y.size(0), (
        "router_logits_x and router_logits_y must have the same first dimension"
    )
    assert 0 <= num_experts_per_token <= num_expert_sqrt * num_expert_sqrt, (
        f"num_experts_per_token should be in [0, {num_expert_sqrt * num_expert_sqrt}], but got {num_experts_per_token}"
    )
    assert num_experts_per_token <= 128, (
        "num_experts_per_token should be less than or equal to 128 for efficiency"
    )
    assert num_experts_per_token % 2 == 0, (
        "num_experts_per_token should be a multiple of 2 for efficiency"
    )


ROUTER_FWD_AUTOTUNE_KEYS = [
    "num_tokens",
    "topk",
    "num_expert_sqrt",
    "num_experts_per_token",
    "topk_square",
]


ROUTER_BWD_AUTOTUNE_KEYS = ["num_tokens", "num_expert_sqrt", "num_experts_per_token"]


EXPERT_FWD_SCORES_TAIL_AUTOTUNE_KEYS = ["num_tokens", "hidden_size"]


EXPERT_FWD_STATES_TAIL_AUTOTUNE_KEYS = ["num_tokens", "hidden_size"]


EXPERT_FWD_SCORES_GROUP_AUTOTUNE_KEYS = ["hidden_size", "group_size", "hidden_size"]


EXPERT_FWD_STATES_GROUP_AUTOTUNE_KEYS = ["hidden_size", "group_size", "hidden_size"]


EXPERT_BWD_STATES_TAIL_AUTOTUNE_KEYS = ["num_tokens", "hidden_size"]


EXPERT_BWD_SCORES_TAIL_AUTOTUNE_KEYS = ["num_tokens", "hidden_size"]


EXPERT_BWD_STATES_GROUP_AUTOTUNE_KEYS = ["hidden_size", "group_size", "hidden_size"]


EXPERT_BWD_SCORES_GROUP_AUTOTUNE_KEYS = ["hidden_size", "group_size", "hidden_size"]


def get_router_fwd_autotune_configs():
    """
    Get autotuning configurations for the router forward kernel.

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    BLOCK_M_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    NUM_WARPS_OPTIONS = [2, 4]
    NUM_STAGES_OPTION = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for nw in NUM_WARPS_OPTIONS:
            for ns in NUM_STAGES_OPTION:
                configs.append(
                    triton.Config(
                        {
                            "TILE_M": bm,
                        },
                        num_warps=nw,
                        num_stages=ns,
                    )
                )
    return configs


def get_router_bwd_autotune_configs():
    """
    Get autotuning configurations for the router backward kernel.

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    BLOCK_M_OPTIONS = [32, 64, 128, 256]
    BLOCK_K_OPTIONS = [16, 32, 64]
    NUM_WARPS_OPTIONS = [2, 4]
    NUM_STAGES_OPTION = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for bk in BLOCK_K_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTION:
                    configs.append(
                        triton.Config(
                            {
                                "TILE_M": bm,
                                "TILE_K": bk,
                            },
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def get_expert_fwd_scores_tail_autotune_configs():
    """
    Get autotuning configurations for the expert forward scores tail kernel.

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    BLOCK_M_OPTIONS = [32, 64, 128, 256]
    BLOCK_N_OPTIONS = [32, 64, 128, 256]
    NUM_WARPS_OPTIONS = [2, 4]
    NUM_STAGES_OPTION = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTION:
                    configs.append(
                        triton.Config(
                            {
                                "TILE_M": bm,
                                "TILE_N": bn,
                            },
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def get_expert_fwd_scores_group_autotune_configs():
    """
    Get autotuning configurations for the expert forward scores group kernel.

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    BLOCK_M_OPTIONS = [32, 64, 128]
    BLOCK_N_OPTIONS = [32, 64, 128]
    BLOCK_K_OPTIONS = [16, 32, 64]
    NUM_WARPS_OPTIONS = [2, 4]
    NUM_STAGES_OPTION = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for bk in BLOCK_K_OPTIONS:
                for nw in NUM_WARPS_OPTIONS:
                    for ns in NUM_STAGES_OPTION:
                        configs.append(
                            triton.Config(
                                {
                                    "TILE_M": bm,
                                    "TILE_N": bn,
                                    "TILE_K": bk,
                                },
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    return configs


def get_expert_fwd_states_tail_autotune_configs():
    """
    Get autotuning configurations for the expert forward states tail kernel.

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    BLOCK_M_OPTIONS = [32, 64, 128, 256]
    BLOCK_N_OPTIONS = [32, 64, 128, 256]
    NUM_WARPS_OPTIONS = [2, 4]
    NUM_STAGES_OPTION = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTION:
                    configs.append(
                        triton.Config(
                            {
                                "TILE_M": bm,
                                "TILE_N": bn,
                            },
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def get_expert_fwd_states_group_autotune_configs():
    """
    Get autotuning configurations for the expert forward states group kernel.

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    BLOCK_M_OPTIONS = [32, 64, 128]
    BLOCK_N_OPTIONS = [32, 64, 128]
    BLOCK_K_OPTIONS = [16, 32, 64]
    NUM_WARPS_OPTIONS = [2, 4]
    NUM_STAGES_OPTION = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for bk in BLOCK_K_OPTIONS:
                for nw in NUM_WARPS_OPTIONS:
                    for ns in NUM_STAGES_OPTION:
                        configs.append(
                            triton.Config(
                                {
                                    "TILE_M": bm,
                                    "TILE_N": bn,
                                    "TILE_K": bk,
                                },
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    return configs


def get_expert_bwd_states_tail_autotune_configs():
    """
    Get autotuning configurations for the expert backward states tail kernel.

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    BLOCK_M_OPTIONS = [32, 64, 128, 256]
    BLOCK_N_OPTIONS = [32, 64, 128, 256]
    NUM_WARPS_OPTIONS = [2, 4]
    NUM_STAGES_OPTION = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTION:
                    configs.append(
                        triton.Config(
                            {
                                "TILE_M": bm,
                                "TILE_N": bn,
                            },
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def get_expert_bwd_scores_tail_autotune_configs():
    """
    Get autotuning configurations for the expert backward scores tail kernel.

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    BLOCK_M_OPTIONS = [32, 64, 128, 256]
    BLOCK_N_OPTIONS = [32, 64, 128, 256]
    NUM_WARPS_OPTIONS = [2, 4]
    NUM_STAGES_OPTION = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTION:
                    configs.append(
                        triton.Config(
                            {
                                "TILE_M": bm,
                                "TILE_N": bn,
                            },
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def get_expert_bwd_states_group_autotune_configs():
    """
    Get autotuning configurations for the expert backward states group kernel.

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    # BLOCK_M_OPTIONS = [32, 64, 128]
    # BLOCK_N_OPTIONS = [32, 64, 128]
    # BLOCK_K_OPTIONS = [16, 32, 64]
    # NUM_WARPS_OPTIONS = [2, 4]
    # NUM_STAGES_OPTION = [1, 2]
    BLOCK_M_OPTIONS = [64]
    BLOCK_N_OPTIONS = [64]
    BLOCK_K_OPTIONS = [16]
    NUM_WARPS_OPTIONS = [4]
    NUM_STAGES_OPTION = [1]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for bk in BLOCK_K_OPTIONS:
                for nw in NUM_WARPS_OPTIONS:
                    for ns in NUM_STAGES_OPTION:
                        configs.append(
                            triton.Config(
                                {
                                    "TILE_M": bm,
                                    "TILE_N": bn,
                                    "TILE_K": bk,
                                },
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    return configs


def get_expert_bwd_scores_group_autotune_configs():
    """
    Get autotuning configurations for the expert backward scores group kernel.

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    # BLOCK_M_OPTIONS = [32, 64, 128]
    # BLOCK_N_OPTIONS = [32, 64, 128]
    # BLOCK_K_OPTIONS = [16, 32, 64]
    # NUM_WARPS_OPTIONS = [2, 4]
    # NUM_STAGES_OPTION = [1, 2]
    BLOCK_M_OPTIONS = [64]
    BLOCK_N_OPTIONS = [64]
    BLOCK_K_OPTIONS = [16]
    NUM_WARPS_OPTIONS = [4]
    NUM_STAGES_OPTION = [1]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for bk in BLOCK_K_OPTIONS:
                for nw in NUM_WARPS_OPTIONS:
                    for ns in NUM_STAGES_OPTION:
                        configs.append(
                            triton.Config(
                                {
                                    "TILE_M": bm,
                                    "TILE_N": bn,
                                    "TILE_K": bk,
                                },
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    return configs
