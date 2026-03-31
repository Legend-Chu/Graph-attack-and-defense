# Graph-attack-and-defense

基于论文 **"Entropy-Based Node Removal for Robust Defense Against Graph Injection Attacks"** 的图注入攻击和防御框架。

## 项目概述

本项目实现了针对社交媒体机器人检测的图注入攻击（IEA）和基于熵的图净化防御（DAEG）。

### 主要贡献

1. **IEA (Influence-Enhanced Attack)**: 基于影响力的图注入攻击
   - 使用局部逆 Hessian 向量近似评估节点影响力
   - 优化连续文本嵌入和离散边生成
   - 在大规模图上保持计算可行性

2. **DAEG (Bots Detect and Eliminate Guard)**: 基于熵的图净化防御
   - 熵感知可疑度评分
   - 邻域不一致性检测
   - 结构感知的图净化
   - 自适应节点移除

## 项目结构

```
Graph-attack-and-defense/
├── IEA.py                 # IEA 攻击模块
├── DAEG.py                # DAEG 防御模块
├── main.py                # 主程序入口
├── utils.py               # 工具函数
├── model.py               # 检测模型定义
├── layer.py               # GNN 层定义
├── cresci-2015/           # Cresci-2015 数据集实验
│   ├── DAEG.py
│   ├── run_IEA.py
│   ├── run_DAGE.py
│   ├── gnia.py
│   └── ...
├── cresci-2017/           # Cresci-2017 数据集实验
├── Twibot-20/             # Twibot-20 数据集实验
├── Twibot-22/             # Twibot-22 数据集实验
└── datasets/              # 数据集目录
```

## 安装依赖

```bash
pip install torch torch_geometric numpy scipy scikit-learn
```

## 使用方法

### 运行完整实验

```bash
# 在 Twibot-22 数据集上运行 IEA 攻击和 DAEG 防御
python main.py --dataset twibot-22 --attack IEA --defense DAEG --injection_rate 0.05

# 在 Cresci-2015 数据集上运行
python main.py --dataset cresci-2015 --attack IEA --defense DAEG --injection_rate 0.01
```

### 单独运行 IEA 攻击

```bash
cd cresci-2015
python run_IEA.py --injection_rate 0.05 --budget 5
```

### 单独运行 DAEG 防御

```bash
cd cresci-2015
python run_DAGE.py
```

## 主要参数

### IEA 攻击参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--injection_rate` | 0.05 | 注入节点比例 |
| `--budget` | 5 | 每个注入节点的边预算 |
| `--lambda_damp` | 0.01 | Hessian 阻尼系数 |
| `--step_size` | 0.1 | 嵌入优化步长 |
| `--gamma` | 0.1 | 相似性惩罚系数 |
| `--k_hop` | 2 | K-hop 邻域大小 |
| `--max_iterations` | 20 | 最大优化迭代次数 |

### DAEG 防御参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--alpha` | 0.6 | 熵权重 |
| `--beta` | 0.4 | 平滑损失权重 |
| `--tau_r` | 1.0 | 可疑度阈值 |
| `--tau_s` | 0.35 | 相似性阈值 |
| `--hidden_dim` | 128 | 隐藏层维度 |
| `--num_epochs` | 100 | 训练轮数 |

## 核心算法

### IEA 攻击

IEA 使用影响力函数来评估注入节点对目标节点的影响：

**影响力分数 (公式 1):**
```
S_IF(v_inj; T) = -1/|T| * Σ_{u∈T} ∇_θ L(u,θ̂)^T * H^{-1}_{θ̂,λ} * ∇_θ L(v_inj,θ̂)
```

**文本嵌入更新 (公式 2):**
```
z' = z + η * sign(∇_z S_IF(v_inj; T))
```

**边分数 (公式 3):**
```
s(u) = S_IF((v_inj, u); T) - γ * SimPenalty(u)
```

### DAEG 防御

DAEG 结合多种信号来识别可疑节点：

**可疑度分数 (公式 5/6):**
```
r_i = α * H̃(i) + β * (1 - S̄_i) + γ * Δ_deg(i)
```

**节点移除规则 (公式 8):**
```
remove(i) = I[r_i > τ_r ∧ Sim̄(i, N(i)) < τ_s]
```

## 数据集

| 数据集 | 用户数 | Bot数 | 边数 |
|--------|--------|-------|------|
| Cresci-2015 | 5,301 | 3,351 | 7,086,134 |
| Cresci-2017 | 14,368 | 10,894 | 6,637,616 |
| Twibot-20 | 8,953,309 | 8,723,736 | 33,716,171 |
| Twibot-22 | 1,000,000 | 139,943 | 170,185,937 |

## 实验结果

在 Twibot-22 数据集上，5% 节点注入攻击下的性能：

| 模型 | Accuracy | F1-macro |
|------|----------|----------|
| GCN | 57.55 | 57.81 |
| R-GCN | 66.21 | 61.21 |
| BotRGCN | 70.13 | 72.44 |
| **DAEG** | **81.21** | **75.83** |

## 引用

如果您使用了本项目的代码，请引用：

```bibtex
@article{chu2024entropy,
  title={Entropy-Based Node Removal for Robust Defense Against Graph Injection Attacks},
  author={Chu, Chuanqi and Zhang, Wei and Zhou, Mingyang and Ao, Xiang and Liao, Hao and Mao, Rui},
  journal={IEEE Transactions on Computational Social Systems},
  year={2024}
}
```

## 许可证

MIT License
