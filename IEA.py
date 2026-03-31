"""
Influence-Enhanced Attack (IEA) - 基于影响力的图注入攻击

基于论文 "Entropy-Based Node Removal for Robust Defense Against Graph Injection Attacks"

IEA 使用局部逆 Hessian 向量近似来评估节点影响力，并据此优化注入节点的特征和边连接。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph, k_hop_subgraph


class InfluenceEvaluator(nn.Module):
    """
    影响力评估器 - 使用局部逆 Hessian 向量近似

    基于论文公式 (1):
    S_IF(v_inj; T) = -1/|T| * sum_{u in T} grad_theta L(u, theta)^T * H^{-1}_{theta, lambda} * grad_theta L(v_inj, theta)

    其中 H_{theta, lambda} = grad^2_theta L_train(theta) + lambda * I 是阻尼 Hessian
    """

    def __init__(self, surrogate_model, lambda_damp=0.01, num_hessian_steps=10):
        super(InfluenceEvaluator, self).__init__()
        self.surrogate_model = surrogate_model
        self.lambda_damp = lambda_damp  # 阻尼系数
        self.num_hessian_steps = num_hessian_steps  # Hessian-vector 乘积的迭代次数

    def compute_gradients(self, x, edge_index, node_idx, labels=None):
        """
        计算给定节点的损失梯度
        """
        self.surrogate_model.zero_grad()
        out = self.surrogate_model(x, edge_index)

        if labels is not None:
            loss = F.cross_entropy(out[node_idx].unsqueeze(0), labels[node_idx].unsqueeze(0))
        else:
            # 如果没有标签，使用模型预测作为伪标签
            pred = out[node_idx].argmax(dim=-1)
            loss = F.cross_entropy(out[node_idx].unsqueeze(0), pred.unsqueeze(0))

        grads = torch.autograd.grad(loss, self.surrogate_model.parameters(), create_graph=True)
        return torch.cat([g.view(-1) for g in grads])

    def hessian_vector_product(self, grads, x, edge_index, node_idx, v):
        """
        计算 Hessian-vector 乘积

        H * v = grad_{theta} (grad_{theta} L * v)
        """
        grad_vector_product = torch.sum(grads * v)
        hvp = torch.autograd.grad(grad_vector_product, self.surrogate_model.parameters(), retain_graph=True)
        return torch.cat([g.view(-1) for g in hvp])

    def inverse_hessian_vector_product(self, x, edge_index, node_idx, v, num_steps=None):
        """
        使用共轭梯度法近似逆 Hessian-vector 乘积

        这是一个迭代方法，避免显式计算和存储 Hessian 矩阵
        """
        if num_steps is None:
            num_steps = self.num_hessian_steps

        grads = self.compute_gradients(x, edge_index, node_idx)

        # 使用阻尼共轭梯度法
        # 初始化
        H_v = self.hessian_vector_product(grads, x, edge_index, node_idx, v)
        b = v  # 右侧向量
        r = b - (H_v + self.lambda_damp * v)  # 残差
        p = r  # 搜索方向
        x_result = torch.zeros_like(v)

        r_dot_r = torch.dot(r, r)

        for _ in range(num_steps):
            H_p = self.hessian_vector_product(grads, x, edge_index, node_idx, p)
            Ap = H_p + self.lambda_damp * p

            alpha = r_dot_r / (torch.dot(p, Ap) + 1e-8)
            x_result = x_result + alpha * p
            r = r - alpha * Ap

            r_dot_r_new = torch.dot(r, r)
            if r_dot_r_new < 1e-10:
                break

            beta = r_dot_r_new / (r_dot_r + 1e-8)
            p = r + beta * p
            r_dot_r = r_dot_r_new

        return x_result

    def compute_influence_score(self, x, edge_index, injected_node_idx, target_nodes, labels=None):
        """
        计算注入节点对目标节点集合的影响力分数

        基于公式 (1):
        S_IF(v_inj; T) = -1/|T| * sum_{u in T} grad_L(u)^T * H^{-1} * grad_L(v_inj)
        """
        # 计算注入节点的梯度
        grad_inj = self.compute_gradients(x, edge_index, injected_node_idx, labels)

        total_influence = 0.0

        for target_idx in target_nodes:
            # 计算目标节点的梯度
            grad_target = self.compute_gradients(x, edge_index, target_idx, labels)

            # 计算逆 Hessian-vector 乘积
            inv_hvp = self.inverse_hessian_vector_product(x, edge_index, target_idx, grad_inj)

            # 计算影响力分数
            influence = -torch.dot(grad_target, inv_hvp)
            total_influence += influence.item()

        return total_influence / len(target_nodes)


class TextEmbeddingOptimizer(nn.Module):
    """
    文本嵌入优化器

    基于论文公式 (2):
    z' = z + eta * sign(grad_z S_IF(v_inj; T))
    """

    def __init__(self, embedding_dim, step_size=0.1):
        super(TextEmbeddingOptimizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.step_size = step_size

    def forward(self, z, influence_grad):
        """
        更新文本嵌入

        Args:
            z: 当前文本嵌入 [embedding_dim]
            influence_grad: 影响力分数对嵌入的梯度

        Returns:
            z': 更新后的文本嵌入
        """
        # 使用符号梯度进行更新
        z_updated = z + self.step_size * torch.sign(influence_grad)

        # 可选：对嵌入进行归一化
        z_updated = F.normalize(z_updated, dim=-1)

        return z_updated


class EdgeSelector(nn.Module):
    """
    边选择器

    基于论文公式 (3):
    s(u) = S_IF((v_inj, u); T) - gamma * SimPenalty(u)

    使用贪婪策略选择 top-B 候选邻居
    """

    def __init__(self, gamma=0.1, budget=5):
        super(EdgeSelector, self).__init__()
        self.gamma = gamma  # 相似性惩罚系数
        self.budget = budget  # 边预算

    def compute_similarity_penalty(self, node_features, candidate_idx, injected_feat):
        """
        计算相似性惩罚

        确保注入节点在统计上与正常用户保持相似
        """
        candidate_feat = node_features[candidate_idx]
        similarity = F.cosine_similarity(injected_feat.unsqueeze(0), candidate_feat, dim=-1)
        # 相似性惩罚：与正常用户越相似，惩罚越大（因为我们不想让注入节点太显眼）
        # 或者：与正常用户越不相似，惩罚越大（因为我们想让注入节点看起来正常）
        # 根据论文，应该是让注入节点保持与正常用户的相似性
        return similarity

    def forward(self, influence_scores, node_features, candidate_indices, injected_feat):
        """
        选择最优的边连接

        Args:
            influence_scores: 候选节点的影响力分数 [num_candidates]
            node_features: 所有节点的特征 [num_nodes, feat_dim]
            candidate_indices: 候选节点索引
            injected_feat: 注入节点的特征

        Returns:
            selected_indices: 选中的候选节点索引
            edge_scores: 边分数
        """
        # 计算相似性惩罚
        sim_penalty = self.compute_similarity_penalty(node_features, candidate_indices, injected_feat)

        # 计算最终边分数：影响力 - 惩罚
        edge_scores = influence_scores - self.gamma * sim_penalty

        # 贪婪选择 top-B
        top_k = min(self.budget, len(candidate_indices))
        _, selected_indices = torch.topk(edge_scores, top_k)

        return candidate_indices[selected_indices], edge_scores


class IEA(nn.Module):
    """
    Influence-Enhanced Attack (IEA) - 影响力增强的图注入攻击

    基于论文 Algorithm 1 实现

    主要步骤：
    1. 获取代理模型预测
    2. 使用 GNIA 初始化注入节点
    3. 迭代优化：
       a. 构建目标节点的 K-hop 自我图
       b. 在自我图上近似逆 Hessian-vector 乘积
       c. 更新连续文本嵌入（公式2）
       d. 计算候选邻居的边分数（公式3）
       e. 贪婪选择 top-B 候选邻居
    4. 返回修改后的图
    """

    def __init__(self, surrogate_model, feat_dim, num_classes=2,
                 lambda_damp=0.01, num_hessian_steps=10,
                 step_size=0.1, gamma=0.1, budget=5,
                 k_hop=2, top_n_candidates=128,
                 max_iterations=20, tolerance=1e-4,
                 device='cuda'):
        super(IEA, self).__init__()

        self.surrogate_model = surrogate_model
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.k_hop = k_hop
        self.top_n_candidates = top_n_candidates
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.device = device
        self.budget = budget

        # 初始化组件
        self.influence_evaluator = InfluenceEvaluator(
            surrogate_model, lambda_damp, num_hessian_steps
        )
        self.text_optimizer = TextEmbeddingOptimizer(feat_dim, step_size)
        self.edge_selector = EdgeSelector(gamma, budget)

    def get_k_hop_ego_graph(self, node_idx, edge_index, num_nodes):
        """
        获取目标节点的 K-hop 自我图
        """
        subset, edge_index_sub, mapping, _ = k_hop_subgraph(
            node_idx, self.k_hop, edge_index, num_nodes=num_nodes
        )
        return subset, edge_index_sub

    def sample_candidates(self, x, edge_index, target_nodes, num_nodes):
        """
        从目标节点的 K-hop 邻域中采样候选节点
        """
        all_candidates = set()
        for target in target_nodes:
            subset, _ = self.get_k_hop_ego_graph(target, edge_index, num_nodes)
            all_candidates.update(subset.tolist())

        # 移除目标节点本身
        candidates = list(all_candidates - set(target_nodes))

        # 限制候选数量
        if len(candidates) > self.top_n_candidates:
            candidates = np.random.choice(candidates, self.top_n_candidates, replace=False).tolist()

        return torch.tensor(candidates, dtype=torch.long, device=self.device)

    def initialize_injected_node(self, x, edge_index, target_nodes, labels=None):
        """
        初始化注入节点（使用简单的方法或 GNIA）
        """
        # 计算目标节点的平均特征作为初始特征
        target_feats = x[target_nodes]
        injected_feat = target_feats.mean(dim=0)

        # 添加小随机噪声
        noise = torch.randn_like(injected_feat) * 0.01
        injected_feat = injected_feat + noise

        return injected_feat

    def forward(self, x, edge_index, target_nodes, labels=None, num_inject=1):
        """
        执行 IEA 攻击

        Args:
            x: 节点特征 [num_nodes, feat_dim]
            edge_index: 边索引 [2, num_edges]
            target_nodes: 目标节点索引
            labels: 节点标签（可选）
            num_inject: 要注入的节点数量

        Returns:
            attacked_x: 攻击后的节点特征
            attacked_edge_index: 攻击后的边索引
            injected_nodes_info: 注入节点信息
        """
        num_nodes = x.size(0)
        injected_nodes_info = []

        for _ in range(num_inject):
            # 初始化注入节点特征
            injected_feat = self.initialize_injected_node(x, edge_index, target_nodes, labels)

            # 采样候选节点
            candidate_indices = self.sample_candidates(x, edge_index, target_nodes, num_nodes)

            # 计算候选节点的影响力分数
            influence_scores = torch.zeros(len(candidate_indices), device=self.device)

            # 临时添加注入节点用于计算影响力
            temp_x = torch.cat([x, injected_feat.unsqueeze(0)], dim=0)
            injected_idx = temp_x.size(0) - 1

            for i, candidate in enumerate(candidate_indices):
                # 构建临时边
                temp_edge = torch.tensor([[injected_idx, candidate.item()],
                                         [candidate.item(), injected_idx]],
                                        dtype=torch.long, device=self.device)
                temp_edge_index = torch.cat([edge_index, temp_edge], dim=1)

                # 计算影响力分数
                score = self.influence_evaluator.compute_influence_score(
                    temp_x, temp_edge_index, injected_idx, target_nodes, labels
                )
                influence_scores[i] = score

            # 选择边
            selected_candidates, edge_scores = self.edge_selector(
                influence_scores, x, candidate_indices, injected_feat
            )

            # 迭代优化文本嵌入
            prev_influence = float('-inf')
            for iteration in range(self.max_iterations):
                # 计算影响力梯度
                injected_feat.requires_grad_(True)
                temp_x = torch.cat([x, injected_feat.unsqueeze(0)], dim=0)

                # 计算总影响力
                total_influence = 0.0
                for target in target_nodes[:10]:  # 限制目标数量以节省计算
                    score = self.influence_evaluator.compute_influence_score(
                        temp_x, edge_index, injected_idx, [target], labels
                    )
                    total_influence += score

                # 检查收敛
                if abs(total_influence - prev_influence) < self.tolerance:
                    break
                prev_influence = total_influence

                # 更新嵌入
                if injected_feat.grad is not None:
                    injected_feat = self.text_optimizer(injected_feat.detach(), injected_feat.grad)
                else:
                    # 如果没有梯度，使用随机扰动
                    injected_feat = injected_feat.detach() + self.text_optimizer.step_size * torch.randn_like(injected_feat) * 0.1

            # 添加注入节点到图中
            x = torch.cat([x, injected_feat.detach().unsqueeze(0)], dim=0)
            new_node_idx = x.size(0) - 1

            # 添加边
            new_edges = torch.stack([
                torch.cat([torch.tensor([new_node_idx], device=self.device), selected_candidates]),
                torch.cat([selected_candidates, torch.tensor([new_node_idx], device=self.device)])
            ], dim=0)
            edge_index = torch.cat([edge_index, new_edges], dim=1)

            injected_nodes_info.append({
                'node_idx': new_node_idx,
                'feature': injected_feat.detach(),
                'connected_nodes': selected_candidates.tolist()
            })

        return x, edge_index, injected_nodes_info


class SurrogateGCN(nn.Module):
    """
    代理 GCN 模型

    用于计算影响力分数的替代模型
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=2):
        super(SurrogateGCN, self).__init__()
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)
        return x


def run_iea_attack(model, x, edge_index, target_nodes, labels=None,
                   num_inject=100, device='cuda', **kwargs):
    """
    运行 IEA 攻击的便捷函数

    Args:
        model: 目标检测模型（用于获取预测）
        x: 节点特征
        edge_index: 边索引
        target_nodes: 目标节点索引
        labels: 节点标签
        num_inject: 要注入的节点数量
        device: 计算设备
        **kwargs: IEA 的其他参数

    Returns:
        attacked_x: 攻击后的特征
        attacked_edge_index: 攻击后的边索引
        injected_info: 注入节点信息
    """
    # 创建代理模型
    surrogate = SurrogateGCN(
        input_dim=x.size(1),
        hidden_dim=kwargs.get('hidden_dim', 64),
        output_dim=2
    ).to(device)

    # 预训练代理模型
    surrogate.train()
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01)

    if labels is not None:
        for epoch in range(20):
            optimizer.zero_grad()
            out = surrogate(x, edge_index)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()

    # 创建 IEA 攻击器
    iea = IEA(
        surrogate_model=surrogate,
        feat_dim=x.size(1),
        device=device,
        **kwargs
    )

    # 执行攻击
    attacked_x, attacked_edge_index, injected_info = iea(
        x, edge_index, target_nodes, labels, num_inject
    )

    return attacked_x, attacked_edge_index, injected_info
