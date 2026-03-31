"""
DAEG (Bots Detect and Eliminate Guard) - 基于熵的图净化防御框架

基于论文 "Entropy-Based Node Removal for Robust Defense Against Graph Injection Attacks"

DAEG 结合熵偏差、邻域不一致性和相似性感知的节点过滤来净化被攻击的图。
它不假设高熵节点一定是恶意的，而是将熵作为一个风险信号，
并结合结构和特征异常来决定节点移除。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, RGCNConv
from torch_geometric.utils import subgraph, to_undirected


class EntropyEvaluator(nn.Module):
    """
    熵评估编码器

    基于论文 Section V-B，计算节点的预测熵和标准化熵偏差

    公式 (4): H(i) = -sum_{k=1}^{C} P(y_k | x_i) * log P(y_k | x_i)
    公式 (5b): tilde{H}(i) = (H(i) - mu_H) / (sigma_H + epsilon)
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_classes=2, dropout=0.3):
        super(EntropyEvaluator, self).__init__()

        # 特征编码层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU()
        )

        # GCN 层用于图结构编码
        self.gcn1 = GCNConv(output_dim, output_dim)
        self.gcn2 = GCNConv(output_dim, output_dim)

        # 分类头
        self.classifier = nn.Linear(output_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]

        Returns:
            logits: 分类 logits [num_nodes, num_classes]
            embeddings: 节点嵌入 [num_nodes, output_dim]
        """
        # 特征编码
        h = self.encoder(x)

        # 图卷积
        h = F.leaky_relu(self.gcn1(h, edge_index))
        h = self.dropout(h)
        h = F.leaky_relu(self.gcn2(h, edge_index))

        # 分类
        logits = self.classifier(h)

        return logits, h

    def compute_entropy(self, logits):
        """
        计算预测熵

        公式 (4): H(i) = -sum_{k} P(y_k | x_i) * log P(y_k | x_i)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def compute_standardized_entropy(self, entropy):
        """
        计算标准化熵偏差

        公式 (5b): tilde{H}(i) = (H(i) - mu_H) / (sigma_H + epsilon)
        """
        mu_H = entropy.mean()
        sigma_H = entropy.std()
        epsilon = 1e-8

        standardized = (entropy - mu_H) / (sigma_H + epsilon)
        return standardized


class NeighborhoodSimilarity(nn.Module):
    """
    邻域相似性计算模块

    基于论文 Section V-C，计算节点与其邻居的平均相似度

    公式 (6): bar{Sim}(i, N(i)) = 1/|N(i)| * sum_{j in N(i)} Sim(u_i, u_j)
    """

    def __init__(self, similarity_type='cosine'):
        super(NeighborhoodSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, node_features, edge_index, node_indices=None):
        """
        计算节点与其邻居的平均相似度

        Args:
            node_features: 节点特征/嵌入 [num_nodes, feat_dim]
            edge_index: 边索引 [2, num_edges]
            node_indices: 要计算的节点索引（None 表示所有节点）

        Returns:
            avg_similarity: 平均邻域相似度 [num_nodes]
        """
        num_nodes = node_features.size(0)

        if node_indices is None:
            node_indices = torch.arange(num_nodes, device=node_features.device)

        # 构建邻接表
        adj_dict = {}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src not in adj_dict:
                adj_dict[src] = []
            if dst not in adj_dict:
                adj_dict[dst] = []
            adj_dict[src].append(dst)
            adj_dict[dst].append(src)

        avg_similarities = torch.zeros(num_nodes, device=node_features.device)

        for node_idx in node_indices:
            neighbors = adj_dict.get(node_idx.item() if isinstance(node_idx, torch.Tensor) else node_idx, [])

            if len(neighbors) == 0:
                avg_similarities[node_idx] = 1.0  # 无邻居时设为高相似度
                continue

            # 计算与邻居的相似度
            node_feat = node_features[node_idx]
            neighbor_feats = node_features[neighbors]

            if self.similarity_type == 'cosine':
                similarities = F.cosine_similarity(
                    node_feat.unsqueeze(0).expand(len(neighbors), -1),
                    neighbor_feats,
                    dim=-1
                )
            elif self.similarity_type == 'jaccard':
                # 对稀疏二值特征使用 Jaccard 相似度
                intersections = (node_feat.unsqueeze(0) * neighbor_feats).sum(dim=-1)
                unions = ((node_feat.unsqueeze(0) + neighbor_feats) > 0).float().sum(dim=-1)
                similarities = intersections / (unions + 1e-8)
            else:
                # 欧氏距离转换为相似度
                distances = torch.norm(node_feat.unsqueeze(0) - neighbor_feats, dim=-1)
                similarities = 1.0 / (1.0 + distances)

            avg_similarities[node_idx] = similarities.mean()

        return avg_similarities


class DegreeDeviation(nn.Module):
    """
    度偏差计算模块

    计算节点度数与局部度分布的偏差
    """

    def __init__(self):
        super(DegreeDeviation, self).__init__()

    def forward(self, edge_index, num_nodes):
        """
        计算每个节点的度偏差

        Args:
            edge_index: 边索引 [2, num_edges]
            num_nodes: 节点数量

        Returns:
            degree_deviation: 度偏差 [num_nodes]
        """
        # 计算度数
        degrees = torch.zeros(num_nodes, device=edge_index.device)
        for i in range(edge_index.size(1)):
            degrees[edge_index[0, i]] += 1
            degrees[edge_index[1, i]] += 1

        # 计算全局统计
        mean_degree = degrees.mean()
        std_degree = degrees.std() + 1e-8

        # 标准化度偏差
        degree_deviation = (degrees - mean_degree) / std_degree

        return degree_deviation


class DAEG(nn.Module):
    """
    DAEG (Bots Detect and Eliminate Guard) - 熵感知图净化框架

    基于论文 Algorithm 2 实现

    主要步骤：
    1. 获取节点预测和预测熵
    2. 计算标准化熵偏差
    3. 编码节点获取增强表示
    4. 计算邻域相似度和度偏差
    5. 计算可疑度分数（公式 5a/6）
    6. 移除满足条件的节点（公式 8）
    7. 在净化后的图上重新训练检测器
    """

    def __init__(self, input_dim, hidden_dim=128, embed_dim=64, num_classes=2,
                 alpha=0.6, beta=0.4, gamma=0.1,
                 tau_r=1.0, tau_s=0.35,
                 dropout=0.3, device='cuda'):
        super(DAEG, self).__init__()

        self.alpha = alpha  # 熵权重
        self.beta = beta    # 相似性权重
        self.gamma = gamma  # 度偏差权重
        self.tau_r = tau_r  # 可疑度阈值
        self.tau_s = tau_s  # 相似性阈值
        self.device = device

        # 熵评估编码器
        self.entropy_evaluator = EntropyEvaluator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout
        )

        # 邻域相似性计算器
        self.neighborhood_similarity = NeighborhoodSimilarity(similarity_type='cosine')

        # 度偏差计算器
        self.degree_deviation = DegreeDeviation()

    def compute_suspiciousness_score(self, standardized_entropy, avg_similarity, degree_dev):
        """
        计算可疑度分数

        公式 (5a/6): r_i = alpha * tilde{H}(i) + beta * (1 - bar{S}_i) + gamma * Delta_deg(i)
        """
        score = self.alpha * standardized_entropy + \
                self.beta * (1 - avg_similarity) + \
                self.gamma * degree_dev
        return score

    def should_remove_node(self, suspiciousness_score, avg_similarity):
        """
        判断是否应该移除节点

        公式 (8): remove(i) = I[r_i > tau_r AND bar{Sim}(i, N(i)) < tau_s]
        """
        remove_mask = (suspiciousness_score > self.tau_r) & (avg_similarity < self.tau_s)
        return remove_mask

    def forward(self, x, edge_index, return_details=False):
        """
        执行图净化

        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            return_details: 是否返回详细信息

        Returns:
            purified_x: 净化后的节点特征
            purified_edge_index: 净化后的边索引
            removed_nodes: 被移除的节点索引
        """
        num_nodes = x.size(0)

        # Step 1: 获取节点预测和预测熵
        logits, embeddings = self.entropy_evaluator(x, edge_index)
        entropy = self.entropy_evaluator.compute_entropy(logits)

        # Step 2: 计算标准化熵偏差
        standardized_entropy = self.entropy_evaluator.compute_standardized_entropy(entropy)

        # Step 3: 编码节点获取增强表示（已在前向传播中完成）
        enhanced_repr = embeddings

        # Step 4 & 5: 计算邻域相似度和度偏差
        avg_similarity = self.neighborhood_similarity(enhanced_repr, edge_index)
        degree_dev = self.degree_deviation(edge_index, num_nodes)

        # Step 6: 计算可疑度分数
        suspiciousness_score = self.compute_suspiciousness_score(
            standardized_entropy, avg_similarity, degree_dev
        )

        # Step 7: 确定要移除的节点
        remove_mask = self.should_remove_node(suspiciousness_score, avg_similarity)
        removed_nodes = torch.where(remove_mask)[0]

        # Step 8: 移除节点和相关边
        keep_mask = ~remove_mask
        keep_indices = torch.where(keep_mask)[0]

        # 创建节点映射（旧索引 -> 新索引）
        node_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=self.device)
        node_mapping[keep_indices] = torch.arange(len(keep_indices), device=self.device)

        # 过滤特征
        purified_x = x[keep_indices]

        # 过滤边
        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        filtered_edges = edge_index[:, edge_mask]
        purified_edge_index = node_mapping[filtered_edges]

        if return_details:
            details = {
                'entropy': entropy,
                'standardized_entropy': standardized_entropy,
                'avg_similarity': avg_similarity,
                'degree_deviation': degree_dev,
                'suspiciousness_score': suspiciousness_score,
                'logits': logits,
                'embeddings': embeddings
            }
            return purified_x, purified_edge_index, removed_nodes, details

        return purified_x, purified_edge_index, removed_nodes

    def compute_loss(self, x, edge_index, labels, labeled_mask):
        """
        计算训练损失

        公式 (7):
        L_EN = alpha * L_cls + beta * L_smooth
        L_cls = -1/|V_L| * sum_{i in V_L} sum_{c=1}^{M} y_ic * log p_ic
        L_smooth = 1/|E| * sum_{(i,j) in E} ||h_i^L - h_j^L||_2^2
        """
        logits, embeddings = self.entropy_evaluator(x, edge_index)

        # 分类损失
        cls_loss = F.cross_entropy(logits[labeled_mask], labels[labeled_mask])

        # 平滑损失
        src, dst = edge_index
        smooth_loss = torch.norm(embeddings[src] - embeddings[dst], dim=-1).mean()

        # 总损失
        total_loss = self.alpha * cls_loss + self.beta * smooth_loss

        return total_loss, cls_loss, smooth_loss


class DAEGDetector(nn.Module):
    """
    完整的 DAEG 检测器

    包含图净化和机器人检测两个阶段
    """

    def __init__(self, input_dim, hidden_dim=128, embed_dim=64, num_classes=2,
                 detector_hidden=64, alpha=0.6, beta=0.4, gamma=0.1,
                 tau_r=1.0, tau_s=0.35, dropout=0.3, device='cuda'):
        super(DAEGDetector, self).__init__()

        self.device = device

        # DAEG 净化模块
        self.daeg = DAEG(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            tau_r=tau_r,
            tau_s=tau_s,
            dropout=dropout,
            device=device
        )

        # 检测器（在净化后的图上工作）
        self.detector = nn.Sequential(
            nn.Linear(input_dim, detector_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(detector_hidden, detector_hidden),
            nn.LeakyReLU(),
            nn.Linear(detector_hidden, num_classes)
        )

        self.gcn_detector = GCNConv(input_dim, detector_hidden)

    def forward(self, x, edge_index, purify=True):
        """
        完整的检测流程

        Args:
            x: 节点特征
            edge_index: 边索引
            purify: 是否进行图净化

        Returns:
            logits: 检测结果
            purified_info: 净化信息
        """
        if purify:
            # 图净化
            purified_x, purified_edge_index, removed_nodes = self.daeg(x, edge_index)

            # 更新节点映射
            num_original = x.size(0)
            keep_mask = torch.ones(num_original, dtype=torch.bool, device=self.device)
            keep_mask[removed_nodes] = False

            # 在净化后的图上进行检测
            h = F.leaky_relu(self.gcn_detector(purified_x, purified_edge_index))
            logits = self.detector(purified_x)

            # 将结果映射回原始节点
            full_logits = torch.zeros(num_original, logits.size(1), device=self.device)
            full_logits[keep_mask] = logits

            # 被移除的节点标记为机器人（标签1）
            full_logits[removed_nodes] = torch.tensor([0.0, 1.0], device=self.device)

            purified_info = {
                'removed_nodes': removed_nodes,
                'keep_mask': keep_mask,
                'purified_x': purified_x,
                'purified_edge_index': purified_edge_index
            }

            return full_logits, purified_info
        else:
            # 不进行净化，直接检测
            h = F.leaky_relu(self.gcn_detector(x, edge_index))
            logits = self.detector(x)
            return logits, {}


def compute_metrics(predictions, labels, injected_nodes=None):
    """
    计算评估指标

    Args:
        predictions: 预测标签
        labels: 真实标签
        injected_nodes: 注入节点索引（用于计算 IR）

    Returns:
        metrics: 包含各种指标的字典
    """
    # 基本指标
    accuracy = (predictions == labels).float().mean().item()

    # True Positive, False Positive, True Negative, False Negative
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    tn = ((predictions == 0) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()

    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

    # 如果提供了注入节点信息，计算 Injected-node Recall (IR)
    if injected_nodes is not None and len(injected_nodes) > 0:
        injected_detected = (predictions[injected_nodes] == 1).sum().item()
        ir = injected_detected / len(injected_nodes)
        metrics['ir'] = ir

    return metrics


def run_daeg_defense(model, x, edge_index, labels=None, labeled_mask=None,
                     num_epochs=100, lr=0.001, device='cuda', **kwargs):
    """
    运行 DAEG 防御的便捷函数

    Args:
        model: 预训练的检测模型（可选）
        x: 节点特征
        edge_index: 边索引
        labels: 节点标签
        labeled_mask: 有标签的节点掩码
        num_epochs: 训练轮数
        lr: 学习率
        device: 计算设备
        **kwargs: DAEG 的其他参数

    Returns:
        purified_x: 净化后的特征
        purified_edge_index: 净化后的边索引
        removed_nodes: 被移除的节点
        detector: 训练好的检测器
    """
    # 创建 DAEG 检测器
    daeg_detector = DAEGDetector(
        input_dim=x.size(1),
        device=device,
        **kwargs
    ).to(device)

    # 如果有标签，训练熵评估器
    if labels is not None and labeled_mask is not None:
        optimizer = torch.optim.Adam(daeg_detector.daeg.entropy_evaluator.parameters(), lr=lr)

        daeg_detector.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            total_loss, cls_loss, smooth_loss = daeg_detector.daeg.compute_loss(
                x, edge_index, labels, labeled_mask
            )
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}, '
                      f'Cls: {cls_loss.item():.4f}, Smooth: {smooth_loss.item():.4f}')

    # 执行图净化
    daeg_detector.eval()
    with torch.no_grad():
        logits, purified_info = daeg_detector(x, edge_index, purify=True)

    return (
        purified_info['purified_x'],
        purified_info['purified_edge_index'],
        purified_info['removed_nodes'],
        daeg_detector
    )
