# Omni Router

The **omni router** is a accelerated primitive for computing the top‑k pairwise routing scores between two sets of logits. It is designed for high‑throughput expert routing or key–value selection in Mixture‑of‑Experts style architectures.

For each token, the router takes two 1D logit vectors of length $N$ and forms all $N^2$ pairwise combinations, then returns the top‑k scores and their indices. The computation is fused in a single GPU kernel that:

- materializes the pairwise scores implicitly inside the Triton kernel,
- finds the top‑k pairs per token,
- writes out only the top‑k scores and flattened pair indices.


## Kernel Interface

The primary user‑facing API is the autograd‑aware wrapper:

```python
from omni_moe.ops.triton import triton_omni_router_func

scores, indices = triton_omni_router_func(router_logits_x, router_logits_y, num_expert_sqrt, num_experts_per_token)
```

**Parameters**

- `router_logits_x` (`torch.Tensor`):
	- shape: `(num_tokens, num_expert_sqrt)`
	- dtype: typically `torch.float32` or `torch.float16` or `torch.bfloat16` (internally cast to `float32`)
	- device: CUDA tensor
- `router_logits_y` (`torch.Tensor`):
	- shape: `(num_tokens, num_expert_sqrt)`
	- dtype: typically `torch.float32` or `torch.float16` or `torch.bfloat16` (internally cast to `float32`)
	- device: CUDA tensor
- `num_expert_sqrt` (`int`):
	- square root of the number of experts; must equal `router_logits_x.size(-1)` and `router_logits_y.size(-1)`
- `num_experts_per_token` (`int`):
	- number of pairwise combinations to select per token, must satisfy `0 <= num_experts_per_token <= num_expert_sqrt * num_expert_sqrt`

**Returns**

- `scores` (`torch.Tensor`):
	- shape: `(num_tokens, num_experts_per_token)`
	- dtype: same as input `router_logits_x` and `router_logits_y` (the kernel runs in `float32` and casts back)
- `indices` (`torch.IntTensor`):
	- shape: `(num_tokens, num_experts_per_token)`
	- flattened pair indices in the range `[0, num_expert_sqrt * num_expert_sqrt)`


## Testing

Router tests and benchmarks live in `tests/test_router.py`. They provide both a pure‑Python reference implementation and Triton‑based implementations to compare against.

To run the router benchmarks on a CUDA‑enabled machine:

```bash
pytest tests/test_router.py -s
```

You can run individual tests with, for example:

```bash
pytest tests/test_router.py::test_router_forward_throughput -s
pytest tests/test_router.py::test_router_backward_throughput -s
```

Make sure that:

- PyTorch is installed with CUDA support,
- Triton is installed and compatible with your CUDA/PyTorch version,
- the environment has a GPU and sufficient memory for the chosen `(num_tokens, num_expert_sqrt, num_experts_per_token)` settings.
