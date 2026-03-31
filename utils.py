"""
工具函数模块

包含数据加载、评估指标、实验配置等功能
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(dataset_name, data_path='./datasets'):
    """
    加载数据集

    Args:
        dataset_name: 数据集名称 ('cresci-2015', 'cresci-2017', 'twibot-20', 'twibot-22')
        data_path: 数据集路径

    Returns:
        data: 包含特征、边、标签等信息的字典
    """
    dataset_path = os.path.join(data_path, dataset_name)

    # 尝试加载预处理好的数据
    data_file = os.path.join(dataset_path, 'processed_data.pt')
    if os.path.exists(data_file):
        return torch.load(data_file)

    # 否则返回 None，需要预处理
    return None


def save_dataset(data, dataset_name, data_path='./datasets'):
    """保存处理后的数据集"""
    dataset_path = os.path.join(data_path, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    data_file = os.path.join(dataset_path, 'processed_data.pt')
    torch.save(data, data_file)


def split_dataset(num_nodes, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    划分数据集

    Args:
        num_nodes: 节点总数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        train_mask, val_mask, test_mask: 数据集划分掩码
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    np.random.seed(seed)
    indices = np.random.permutation(num_nodes)

    train_end = int(num_nodes * train_ratio)
    val_end = train_end + int(num_nodes * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


def compute_evaluation_metrics(y_true, y_pred, y_prob=None):
    """
    计算评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（用于计算 AP）

    Returns:
        metrics: 指标字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics['ap'] = average_precision_score(y_true, y_prob, average='macro')
        except:
            metrics['ap'] = 0.0

    return metrics


def compute_attack_success_rate(target_nodes, original_preds, attacked_preds, target_labels):
    """
    计算攻击成功率 (ASR)

    ASR = 成功改变预测的目标节点数 / 总目标节点数

    Args:
        target_nodes: 目标节点索引
        original_preds: 攻击前的预测
        attacked_preds: 攻击后的预测
        target_labels: 目标节点的真实标签

    Returns:
        asr: 攻击成功率
    """
    # 原本被正确分类的节点
    correctly_classified = original_preds[target_nodes] == target_labels[target_nodes]

    # 攻击后被错误分类的节点
    misclassified_after_attack = attacked_preds[target_nodes] != target_labels[target_nodes]

    # 成功的攻击：原本正确分类但攻击后被错误分类
    successful_attacks = correctly_classified & misclassified_after_attack

    # ASR
    num_successful = successful_attacks.sum().item()
    num_vulnerable = correctly_classified.sum().item()

    if num_vulnerable == 0:
        return 0.0

    asr = num_successful / num_vulnerable
    return asr


def compute_injected_node_recall(removed_nodes, injected_nodes):
    """
    计算注入节点召回率 (IR)

    IR = 被正确识别的注入节点数 / 总注入节点数

    Args:
        removed_nodes: 被移除的节点索引
        injected_nodes: 注入节点索引

    Returns:
        ir: 注入节点召回率
    """
    if len(injected_nodes) == 0:
        return 0.0

    removed_set = set(removed_nodes.tolist() if isinstance(removed_nodes, torch.Tensor) else removed_nodes)
    injected_set = set(injected_nodes.tolist() if isinstance(injected_nodes, torch.Tensor) else injected_nodes)

    correctly_removed = len(removed_set & injected_set)
    ir = correctly_removed / len(injected_set)

    return ir


def compute_false_removal_rate(removed_nodes, injected_nodes, num_nodes):
    """
    计算错误移除率 (FRR)

    FRR = 被错误移除的正常节点数 / 总正常节点数

    Args:
        removed_nodes: 被移除的节点索引
        injected_nodes: 注入节点索引
        num_nodes: 节点总数

    Returns:
        frr: 错误移除率
    """
    removed_set = set(removed_nodes.tolist() if isinstance(removed_nodes, torch.Tensor) else removed_nodes)
    injected_set = set(injected_nodes.tolist() if isinstance(injected_nodes, torch.Tensor) else injected_nodes)

    num_normal_nodes = num_nodes - len(injected_set)
    if num_normal_nodes == 0:
        return 0.0

    wrongly_removed = len(removed_set - injected_set)
    frr = wrongly_removed / num_normal_nodes

    return frr


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience: 容忍的轮数
            min_delta: 最小改进
            mode: 'max' 或 'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, log_path='./logs', exp_name='experiment'):
        self.log_path = log_path
        self.exp_name = exp_name
        os.makedirs(log_path, exist_ok=True)
        self.log_file = os.path.join(log_path, f'{exp_name}.json')
        self.logs = []

    def log(self, epoch, metrics, phase='train'):
        """记录一轮的指标"""
        entry = {
            'epoch': epoch,
            'phase': phase,
            'metrics': metrics
        }
        self.logs.append(entry)

    def save(self):
        """保存日志到文件"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def load(self):
        """加载日志"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)


def get_device():
    """获取计算设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage():
    """获取 GPU 内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return {'allocated_gb': allocated, 'reserved_gb': reserved}
    return {'allocated_gb': 0, 'reserved_gb': 0}


def normalize_features(x, method='standard'):
    """
    特征归一化

    Args:
        x: 特征矩阵
        method: 归一化方法 ('standard', 'minmax', 'l2')

    Returns:
        normalized_x: 归一化后的特征
    """
    if method == 'standard':
        mean = x.mean(dim=0)
        std = x.std(dim=0) + 1e-8
        return (x - mean) / std
    elif method == 'minmax':
        min_val = x.min(dim=0)[0]
        max_val = x.max(dim=0)[0]
        return (x - min_val) / (max_val - min_val + 1e-8)
    elif method == 'l2':
        return F.normalize(x, p=2, dim=-1)
    else:
        return x


def add_self_loops(edge_index, num_nodes):
    """添加自环"""
    loop_index = torch.arange(num_nodes, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index


def to_undirected(edge_index):
    """转换为无向图"""
    src, dst = edge_index
    new_src = torch.cat([src, dst], dim=0)
    new_dst = torch.cat([dst, src], dim=0)
    return torch.stack([new_src, new_dst], dim=0)


def sample_neighbors(edge_index, node_idx, num_samples=10):
    """
    采样节点的邻居

    Args:
        edge_index: 边索引
        node_idx: 节点索引
        num_samples: 采样数量

    Returns:
        neighbor_indices: 邻居索引
    """
    # 找到节点的所有邻居
    mask = edge_index[0] == node_idx
    neighbors = edge_index[1][mask]

    if len(neighbors) > num_samples:
        indices = torch.randperm(len(neighbors))[:num_samples]
        neighbors = neighbors[indices]

    return neighbors


def build_adjacency_list(edge_index, num_nodes):
    """构建邻接表"""
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].append(dst)
    return adj_list


def get_k_hop_neighbors(node_idx, edge_index, k, num_nodes):
    """
    获取 K-hop 邻居

    Args:
        node_idx: 起始节点索引
        edge_index: 边索引
        k: 跳数
        num_nodes: 节点总数

    Returns:
        neighbors: K-hop 邻居集合
    """
    adj_list = build_adjacency_list(edge_index, num_nodes)

    visited = {node_idx}
    current_level = {node_idx}

    for _ in range(k):
        next_level = set()
        for node in current_level:
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.add(neighbor)
        current_level = next_level

    return visited - {node_idx}  # 排除起始节点本身


# 实验配置
EXPERIMENT_CONFIG = {
    'datasets': ['cresci-2015', 'cresci-2017', 'twibot-20', 'twibot-22'],
    'injection_rates': [0.01, 0.02, 0.05, 0.08],
    'attack_methods': ['IEA', 'G-NIA', 'NIPA', 'LP-GIA', 'clean'],
    'defense_methods': ['DAEG', 'BotRGCN', 'BotGCN', 'GCN', 'R-GCN', 'HGT', 'GNNGuard'],

    # IEA 参数
    'iea': {
        'lambda_damp': 0.01,
        'num_hessian_steps': 10,
        'step_size': 0.1,
        'gamma': 0.1,
        'budget': 5,
        'k_hop': 2,
        'top_n_candidates': 128,
        'max_iterations': 20,
        'tolerance': 1e-4
    },

    # DAEG 参数
    'daeg': {
        'hidden_dim': 128,
        'embed_dim': 64,
        'alpha': 0.6,
        'beta': 0.4,
        'gamma': 0.1,
        'tau_r': 1.0,
        'tau_s': 0.35,
        'dropout': 0.3
    },

    # 训练参数
    'training': {
        'lr': 0.001,
        'batch_size': 128,
        'num_epochs': 100,
        'weight_decay': 5e-4,
        'patience': 10
    }
}
