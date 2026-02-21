<div align="center">

![banner](./assets/omni_moe_banner.png)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2602.05711-b31b1b.svg)](https://arxiv.org/abs/2602.05711)
[![Website](https://img.shields.io/badge/website-OmniMoE-yellow.svg)](https://omnimoeexpl-lhzzgsta.manus.space/)

**English** | [简体中文](./README_zh.md)

</div>

## Why OmniMoE

As LLMs scale, dense Transformer blocks become increasingly expensive. MoE improves parameter efficiency by activating only a subset of parameters per token.
However, existing MoE designs often face a trade-off:

- **Coarse-grained experts** are easier to utilize efficiently, but the number of experts is limited, and under a fixed compute budget they may waste compute or lose information.
- **Fine-grained experts** can scale the number of experts dramatically, but are often bottlenecked by routing quality and memory bandwidth, making stable end-to-end gains hard.

OmniMoE advances the standard MoE paradigm by decomposing experts into fine-grained units enabling massive scaling within a fixed compute budget. It addresses the trade-off between routing quality and hardware efficiency through the following innovations:

- **Dynamic Assembled Experts** conceptualizes experts as single neurons for the on-the-fly assembly of precise parameter combinations.
- **Cartesian Product Router** factorizes the router to efficiently index massive expert spaces without prohibitive computational costs.
- **Expert-Centric Scheduling** reorders computation to process tokens in batches per expert maximizing memory locality and arithmetic efficiency.
- **Hybrid Parallel Design** combines a shared dense backbone for general reasoning with sparse fine-grained experts for specialized knowledge retrieval.


## Key Features

- Router kernel: per-token top-k selection over a Cartesian-product expert space
- Expert kernel: fused gather, activation, and weighted accumulation
- MLP kernel: for the shared dense branch


## Installation

### Requirements

- Python >= 3.8
- PyTorch with CUDA support
- Triton
- transformers

### Install from source

```bash
git clone https://github.com/flash-algo/omni-moe.git
cd omni-moe
pip install -e .
```


## Quick Start

### OmniMoE module

```python
import torch

from omni_moe.modules.omni_moe import OmniMoE, OmniMoEConfig

device = torch.device("cuda")
dtype = torch.bfloat16

config = OmniMoEConfig(
	hidden_size=1024,
	intermediate_size=4096,
	hidden_act="silu",
	num_experts=4096,
	num_experts_per_token=16,
)

x = torch.randn(1, 4096, config.hidden_size, device=device, dtype=dtype)
moe = OmniMoE(config).to(device=device, dtype=dtype)

y = moe(x)
print(y.shape)
```


## Benchmarks

Includes pytest-based kernel benchmarks and tests.

- Router: [docs/omni_router.md](docs/omni_router.md), tests in `tests/test_router.py`
- Expert: [docs/omni_expert.md](docs/omni_expert.md), tests in `tests/test_expert.py`

Run all tests:

```bash
pytest -q
```

Run a specific kernel benchmark:

```bash
pytest tests/test_router.py -s
pytest tests/test_expert.py -s
```


## License

See [LICENSE](LICENSE).


## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{shi2026omnimoe,
      title={OmniMoE: An Efficient MoE by Orchestrating Atomic Experts at Scale}, 
      author={Jingze Shi and Zhangyang Peng and Yizhang Zhu and Yifan Wu and Guang Liu and Yuyu Luo},
      year={2026},
      eprint={2602.05711},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.05711}, 
}
```
