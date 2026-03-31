"""
DAEG (Bots Detect and Eliminate Guard) - Cresci-2015 数据集版本

这是 DAEG 防御框架在 Cresci-2015 数据集上的具体实现。
完整实现请参考项目根目录的 DAEG.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入完整的 DAEG 实现
try:
    from DAEG import DAEG, DAEGDetector, run_daeg_defense, compute_metrics
    FULL_DAEG_AVAILABLE = True
except ImportError:
    FULL_DAEG_AVAILABLE = False


class GCN(torch.nn.Module):
    """基础 GCN 模型"""
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class EntropyEvaluator(nn.Module):
    """
    熵评估器 - 用于 Cresci-2015 数据集

    计算节点的预测熵和标准化熵偏差
    """

    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.3):
        super(EntropyEvaluator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU()
        )

        self.gcn1 = GCNConv(hidden_dim // 2, hidden_dim // 2)
        self.gcn2 = GCNConv(hidden_dim // 2, hidden_dim // 2)

        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.encoder(x)
        h = F.leaky_relu(self.gcn1(h, edge_index))
        h = self.dropout(h)
        h = F.leaky_relu(self.gcn2(h, edge_index))
        logits = self.classifier(h)
        return logits, h

    def compute_entropy(self, logits):
        """计算预测熵 H(i) = -sum P(y_k|x_i) * log P(y_k|x_i)"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def compute_standardized_entropy(self, entropy):
        """计算标准化熵偏差 tilde{H}(i) = (H(i) - mu_H) / (sigma_H + epsilon)"""
        mu_H = entropy.mean()
        sigma_H = entropy.std()
        return (entropy - mu_H) / (sigma_H + 1e-8)


class DAEGCresci(nn.Module):
    """
    DAEG for Cresci-2015 dataset

    完整的图净化防御框架
    """

    def __init__(self, input_dim, hidden_dim=128, alpha=0.6, beta=0.4,
                 tau_r=1.0, tau_s=0.35, device='cuda'):
        super(DAEGCresci, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.tau_r = tau_r
        self.tau_s = tau_s
        self.device = device

        self.entropy_evaluator = EntropyEvaluator(input_dim, hidden_dim)

    def compute_neighborhood_similarity(self, embeddings, edge_index):
        """计算邻域平均相似度"""
        num_nodes = embeddings.size(0)
        adj_dict = {}

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src not in adj_dict:
                adj_dict[src] = []
            if dst not in adj_dict:
                adj_dict[dst] = []
            adj_dict[src].append(dst)
            adj_dict[dst].append(src)

        avg_similarities = torch.zeros(num_nodes, device=embeddings.device)

        for node_idx in range(num_nodes):
            neighbors = adj_dict.get(node_idx, [])
            if len(neighbors) == 0:
                avg_similarities[node_idx] = 1.0
                continue

            node_feat = embeddings[node_idx]
            neighbor_feats = embeddings[neighbors]
            similarities = F.cosine_similarity(
                node_feat.unsqueeze(0).expand(len(neighbors), -1),
                neighbor_feats, dim=-1
            )
            avg_similarities[node_idx] = similarities.mean()

        return avg_similarities

    def compute_suspiciousness_score(self, standardized_entropy, avg_similarity):
        """计算可疑度分数 r_i = alpha * tilde{H}(i) + beta * (1 - bar{S}_i)"""
        return self.alpha * standardized_entropy + self.beta * (1 - avg_similarity)

    def forward(self, x, edge_index):
        """执行图净化"""
        num_nodes = x.size(0)

        # 获取预测和熵
        logits, embeddings = self.entropy_evaluator(x, edge_index)
        entropy = self.entropy_evaluator.compute_entropy(logits)
        standardized_entropy = self.entropy_evaluator.compute_standardized_entropy(entropy)

        # 计算邻域相似度
        avg_similarity = self.compute_neighborhood_similarity(embeddings, edge_index)

        # 计算可疑度分数
        suspiciousness_score = self.compute_suspiciousness_score(
            standardized_entropy, avg_similarity
        )

        # 确定要移除的节点
        remove_mask = (suspiciousness_score > self.tau_r) & (avg_similarity < self.tau_s)
        removed_nodes = torch.where(remove_mask)[0]

        # 过滤节点和边
        keep_mask = ~remove_mask
        keep_indices = torch.where(keep_mask)[0]

        node_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=self.device)
        node_mapping[keep_indices] = torch.arange(len(keep_indices), device=self.device)

        purified_x = x[keep_indices]

        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        filtered_edges = edge_index[:, edge_mask]
        purified_edge_index = node_mapping[filtered_edges]

        return purified_x, purified_edge_index, removed_nodes, {
            'entropy': entropy,
            'suspiciousness_score': suspiciousness_score,
            'logits': logits
        }


def influence_function(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, labels):
    """
    使用 GCN 计算节点嵌入用于影响力分析

    Args:
        des_tensor: 描述特征 [num_nodes, 768]
        tweets_tensor: 推文特征 [num_nodes, 768]
        num_prop: 数值特征 [num_nodes, 5]
        category_prop: 类别特征 [num_nodes, 1]
        edge_index: 边索引 [2, num_edges]
        labels: 节点标签

    Returns:
        out: 节点嵌入
    """
    input_dim = des_tensor.size(1) + tweets_tensor.size(1) + num_prop.size(1) + category_prop.size(1)
    hidden_dim = 128

    x = torch.cat((des_tensor, tweets_tensor, num_prop, category_prop), dim=1)
    edge_index = edge_index.to(x.device)

    model = GCN(input_dim, hidden_dim).to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[labels >= 0], labels[labels >= 0])
        loss.backward()
        optimizer.step()

    return out


def remove_injected_nodes(out, labels, threshold=0.5):
    """
    基于相似度识别并移除注入节点

    Args:
        out: 节点嵌入
        labels: 节点标签
        threshold: 相似度阈值

    Returns:
        injected_indices: 识别出的注入节点索引
    """
    sim_matrix = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1)

    bot_indices = torch.where(labels == 1)[0]
    injected_indices = []

    for idx in bot_indices:
        similar_nodes = torch.where(sim_matrix[idx] > threshold)[0]
        injected_indices.extend(similar_nodes.tolist())

    injected_indices = list(set(injected_indices))

    if len(injected_indices) > 1:
        injected_indices = sorted(injected_indices)[:len(injected_indices) // 2]

    return injected_indices


def run_daeg_cresci(des_tensor, tweets_tensor, num_prop, category_prop,
                    edge_index, labels, train_mask, device='cuda'):
    """
    在 Cresci-2015 数据集上运行 DAEG 防御

    Args:
        des_tensor: 描述特征
        tweets_tensor: 推文特征
        num_prop: 数值特征
        category_prop: 类别特征
        edge_index: 边索引
        labels: 标签
        train_mask: 训练集掩码
        device: 计算设备

    Returns:
        purified_data: 净化后的数据
    """
    # 合并特征
    x = torch.cat((des_tensor, tweets_tensor, num_prop, category_prop), dim=1)
    x = x.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)

    # 创建 DAEG 模型
    input_dim = x.size(1)
    daeg = DAEGCresci(input_dim=input_dim, hidden_dim=128, device=device).to(device)

    # 训练熵评估器
    optimizer = torch.optim.Adam(daeg.entropy_evaluator.parameters(), lr=0.001)

    daeg.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits, _ = daeg.entropy_evaluator(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    # 执行图净化
    daeg.eval()
    with torch.no_grad():
        purified_x, purified_edge_index, removed_nodes, details = daeg(x, edge_index)

    print(f'Removed {len(removed_nodes)} suspicious nodes')

    # 分离净化后的特征
    des_size = des_tensor.size(1)
    tweet_size = tweets_tensor.size(1)
    num_size = num_prop.size(1)

    new_des_tensor = purified_x[:, :des_size]
    new_tweets_tensor = purified_x[:, des_size:des_size + tweet_size]
    new_num_prop = purified_x[:, des_size + tweet_size:des_size + tweet_size + num_size]
    new_category_prop = purified_x[:, des_size + tweet_size + num_size:]

    # 创建新的标签（移除被移除节点的标签）
    keep_mask = torch.ones(labels.size(0), dtype=torch.bool, device=device)
    keep_mask[removed_nodes] = False
    new_labels = labels[keep_mask]

    return {
        'des_tensor': new_des_tensor.cpu(),
        'tweets_tensor': new_tweets_tensor.cpu(),
        'num_prop': new_num_prop.cpu(),
        'category_prop': new_category_prop.cpu(),
        'labels': new_labels.cpu(),
        'edge_index': purified_edge_index.cpu(),
        'removed_nodes': removed_nodes.cpu()
    }


# 使用示例
if __name__ == '__main__':
    # 示例数据（实际使用时需要加载真实数据）
    num_nodes = 1000
    des_tensor = torch.randn(num_nodes, 768)
    tweets_tensor = torch.randn(num_nodes, 768)
    num_prop = torch.randn(num_nodes, 5)
    category_prop = torch.randn(num_nodes, 1)
    edge_index = torch.randint(0, num_nodes, (2, 5000))
    labels = torch.randint(0, 2, (num_nodes,))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:700] = True

    # 运行 DAEG
    result = run_daeg_cresci(
        des_tensor, tweets_tensor, num_prop, category_prop,
        edge_index, labels, train_mask
    )

    # 保存结果
    torch.save(result['des_tensor'], 'new_des_tensor.pt')
    torch.save(result['tweets_tensor'], 'new_tweets_tensor.pt')
    torch.save(result['num_prop'], 'new_num_prop.pt')
    torch.save(result['category_prop'], 'new_category_prop.pt')
    torch.save(result['labels'], 'new_labels.pt')
    torch.save(result['edge_index'], 'new_edge_index.pt')
