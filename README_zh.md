<div align="center">

![banner](./assets/omni_moe_banner.png)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2602.05711-b31b1b.svg)](https://arxiv.org/abs/2602.05711)
[![Website](https://img.shields.io/badge/website-OmniMoE-yellow.svg)](https://omnimoeexpl-lhzzgsta.manus.space/)

[English](./README.md) | **简体中文**

</div>

## 为什么 OmniMoE

随着大语言模型规模增长, 稠密 Transformer block 的成本会快速上升. MoE 通过每个 token 仅激活一部分参数提升参数效率.
但现有 MoE 设计常常面临取舍:

- **粗粒度专家** 更容易高效利用算力, 但专家数量受限, 并且固定计算预算下可能会造成算力浪费或丢失信息.
- **细粒度专家** 可以把专家数量扩展到非常大, 但往往受路由质量与显存带宽瓶颈影响, 难以获得稳定的端到端收益.

OmniMoE 通过将专家分解为细粒度单元推进了标准 MoE 范式，从而在固定计算预算内实现大规模扩展。它通过以下创新解决了路由质量和硬件效率之间的权衡：

- **动态组装专家** 将专家视为单个神经元以便即时组装特定于 Token 的精确参数组合
- **笛卡尔积路由** 对路由进行因式分解以在不引入过高计算成本的情况下高效索引海量专家空间
- **专家中心调度** 通过按专家 ID 重排序计算并批量处理 Token 来最大化访存局部性与算术效率
- **混合并行设计** 将用于通用推理的共享稠密骨干网络与用于专业知识检索的稀疏细粒度专家相结合


## 主要特性

- 路由内核: 在笛卡尔积专家空间上做 per-token top-k 选择
- 专家内核: 融合 gather, 激活, 加权累加
- MLP 内核: 用于共享稠密分支


## 安装

### 环境要求

- Python >= 3.8
- 支持 CUDA 的 PyTorch
- Triton
- transformers

### 从源码安装

```bash
git clone https://github.com/flash-algo/omni-moe.git
cd omni-moe
pip install -e .
```


## 快速开始

### OmniMoE 模块

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


## 基准测试

包含基于 pytest 的 kernel 基准与测试.

- Router: 说明见 [docs/omni_router.md](docs/omni_router.md), 测试在 `tests/test_router.py`
- Expert: 说明见 [docs/omni_expert.md](docs/omni_expert.md), 测试在 `tests/test_expert.py`

运行所有测试:

```bash
pytest -q
```

运行单个 kernel benchmark:

```bash
pytest tests/test_router.py -s
pytest tests/test_expert.py -s
```


## License

见 [LICENSE](LICENSE).


## Citation

如果你认为 OmniMoE 对你的研究有帮助, 请考虑引用:

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
