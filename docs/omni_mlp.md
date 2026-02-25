# Omni MLP

The **omni MLP** is an accelerated primitive for computing the SwiGLU (Swish-Gated Linear Unit) activation commonly used in shared dense backbone. It fuses the element-wise gating operation into a single efficient GPU kernel.

For each token, the MLP takes an input hidden state and applies the gate, up, and down projections with a SwiGLU activation in between. The computation is fused in a single GPU kernel that:

- computes the SiLU activation on the gate projections,
- performs element-wise multiplication with the up projections,
- supports efficient backward pass with recomputation to save memory.


## Kernel Interface

The primary user-facing API is the autograd-aware wrapper:

```python
from omni_moe.ops.triton import triton_omni_mlp_func

y = triton_omni_mlp_func(x, gate_weight, up_weight, down_weight)
```

**Parameters**

- `x` (`torch.Tensor`):
	- shape: `(num_tokens, hidden_size)`
	- dtype: typically `torch.float32` or `torch.float16` or `torch.bfloat16`
	- device: CUDA tensor
- `gate_weight` (`torch.Tensor`):
	- shape: `(intermediate_size, hidden_size)`
	- dtype: same as `x`
	- device: CUDA tensor
- `up_weight` (`torch.Tensor`):
	- shape: `(intermediate_size, hidden_size)`
	- dtype: same as `x`
	- device: CUDA tensor
- `down_weight` (`torch.Tensor`):
	- shape: `(hidden_size, intermediate_size)`
	- dtype: same as `x`
	- device: CUDA tensor

**Returns**

- `y` (`torch.Tensor`):
	- shape: `(num_tokens, hidden_size)`
	- dtype: same as input `x`


## Testing

MLP tests and benchmarks live in `tests/test_mlp.py`. They provide both a pure-Python reference implementation and Triton-based implementations to compare against.

To run the MLP benchmarks on a CUDA-enabled machine:

```bash
pytest tests/test_mlp.py -s
```

You can run individual tests with, for example:

```bash
pytest tests/test_mlp.py::test_mlp_forward_throughput -s
pytest tests/test_mlp.py::test_mlp_backward_throughput -s
```

Make sure that:

- PyTorch is installed with CUDA support,
- Triton is installed and compatible with your CUDA/PyTorch version,
- the environment has a GPU and sufficient memory for the chosen `(num_tokens, hidden_size, intermediate_size)` settings.
