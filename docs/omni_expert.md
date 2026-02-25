# Omni Expert

The **omni expert** is an accelerated primitive for Mixture-of-Experts expert computation. It converts token-centric pairs into an expert-centric layout, then runs fused kernels for expert scores and expert states.

For each token, the expert module applies a up projection to compute expert scores, applies the SiLU activation and routing weights, and then applies the down projection to produce the final output. The computation is split into two fused GPU kernels that:

- compute expert scores for each (token, expert) pair,
- apply SiLU and routing weights,
- accumulate expert states back into per-token outputs.


## Kernel Interface

The primary user-facing API is the autograd-aware wrapper:

```python
from omni_moe.ops.triton import triton_omni_expert_func

expert_states = triton_omni_expert_func(hidden_states, up_weights, down_weights, routing_weights, indices)
```

**Parameters**

- `hidden_states` (`torch.Tensor`):
    - shape: `(num_tokens, hidden_size)`
    - dtype: typically `torch.float32` or `torch.float16` or `torch.bfloat16`
    - device: CUDA tensor
- `up_weights` (`torch.Tensor`):
    - shape: `(num_experts, hidden_size)`
    - dtype: same as `hidden_states`
    - device: CUDA tensor
- `down_weights` (`torch.Tensor`):
    - shape: `(num_experts, hidden_size)`
    - dtype: same as `hidden_states`
    - device: CUDA tensor
- `routing_weights` (`torch.Tensor`):
    - shape: `(num_tokens, num_experts_per_token)`
    - dtype: same as `hidden_states`
    - device: CUDA tensor
- `indices` (`torch.Tensor`):
    - shape: `(num_tokens, num_experts_per_token)`
    - dtype: `torch.int32`
    - device: CUDA tensor

**Returns**

- `expert_states` (`torch.Tensor`):
    - shape: `(num_tokens, hidden_size)`
    - dtype: same as `hidden_states`


## Testing

Expert tests and benchmarks live in `tests/test_expert.py`. They provide a PyTorch reference implementation and compare against the Triton implementation.

To run the expert tests on a CUDA-enabled machine:

```bash
pytest tests/test_expert.py -s
```

You can run individual tests with, for example:

```bash
pytest tests/test_expert.py::test_expert_forward_throughput -s
pytest tests/test_expert.py::test_expert_backward_throughput -s
```

Make sure that:

- PyTorch is installed with CUDA support,
- Triton is installed and compatible with your CUDA/PyTorch version,
- the environment has a GPU and sufficient memory for the chosen `(num_tokens, hidden_size, num_experts, num_experts_per_token)` settings.
