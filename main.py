"""
Graph Attack and Defense - 主程序入口

基于论文 "Entropy-Based Node Removal for Robust Defense Against Graph Injection Attacks"

使用方法:
    python main.py --dataset twibot-22 --attack IEA --defense DAEG --injection_rate 0.05

"""

import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from IEA import IEA, SurrogateGCN, run_iea_attack
from DAEG import DAEG, DAEGDetector, run_daeg_defense, compute_metrics
from utils import (
    set_seed, get_device, split_dataset, compute_evaluation_metrics,
    compute_attack_success_rate, compute_injected_node_recall,
    compute_false_removal_rate, EarlyStopping, ExperimentLogger,
    EXPERIMENT_CONFIG
)


def run_attack(args, model, x, edge_index, labels, target_nodes, device):
    """
    运行攻击

    Args:
        args: 命令行参数
        model: 目标检测模型
        x: 节点特征
        edge_index: 边索引
        labels: 节点标签
        target_nodes: 目标节点索引
        device: 计算设备

    Returns:
        attacked_x: 攻击后的特征
        attacked_edge_index: 攻击后的边索引
        injected_info: 注入节点信息
    """
    print(f"\n{'='*50}")
    print(f"Running {args.attack} attack...")
    print(f"{'='*50}")

    num_inject = int(x.size(0) * args.injection_rate)
    print(f"Number of nodes to inject: {num_inject}")

    start_time = time.time()

    if args.attack == 'IEA':
        # 使用 IEA 攻击
        attacked_x, attacked_edge_index, injected_info = run_iea_attack(
            model=model,
            x=x,
            edge_index=edge_index,
            target_nodes=target_nodes,
            labels=labels,
            num_inject=num_inject,
            device=device,
            lambda_damp=args.lambda_damp,
            step_size=args.step_size,
            gamma=args.gamma,
            budget=args.budget,
            k_hop=args.k_hop,
            max_iterations=args.max_iterations
        )
    elif args.attack == 'G-NIA':
        # 使用 G-NIA 攻击 (简化版本)
        from gnia import GNIA
        # 这里需要适配 G-NIA 的接口
        # 暂时使用简化版本
        attacked_x, attacked_edge_index, injected_info = simple_injection_attack(
            x, edge_index, num_inject, device
        )
    elif args.attack == 'clean':
        # 无攻击，注入零特征节点
        attacked_x, attacked_edge_index, injected_info = clean_injection(
            x, edge_index, num_inject, device
        )
    else:
        raise ValueError(f"Unknown attack method: {args.attack}")

    attack_time = time.time() - start_time
    print(f"Attack completed in {attack_time:.2f} seconds")
    print(f"Injected {len(injected_info)} nodes")

    return attacked_x, attacked_edge_index, injected_info


def simple_injection_attack(x, edge_index, num_inject, device):
    """
    简单的注入攻击（作为基线）

    随机生成节点特征和边连接
    """
    feat_dim = x.size(1)
    num_nodes = x.size(0)

    injected_info = []
    new_x = x.clone()
    new_edge_index = edge_index.clone()

    for i in range(num_inject):
        # 生成注入节点特征（基于正常节点的统计分布）
        mean_feat = x.mean(dim=0)
        std_feat = x.std(dim=0)
        injected_feat = mean_feat + std_feat * torch.randn(feat_dim, device=device) * 0.1

        # 添加节点
        new_x = torch.cat([new_x, injected_feat.unsqueeze(0)], dim=0)
        new_node_idx = new_x.size(0) - 1

        # 随机选择邻居
        num_neighbors = np.random.randint(1, 6)
        neighbors = torch.randperm(num_nodes)[:num_neighbors].to(device)

        # 添加边
        new_edges = torch.stack([
            torch.cat([torch.tensor([new_node_idx], device=device).expand(num_neighbors), neighbors]),
            torch.cat([neighbors, torch.tensor([new_node_idx], device=device).expand(num_neighbors)])
        ], dim=0)
        new_edge_index = torch.cat([new_edge_index, new_edges], dim=1)

        injected_info.append({
            'node_idx': new_node_idx,
            'feature': injected_feat,
            'connected_nodes': neighbors.tolist()
        })

    return new_x, new_edge_index, injected_info


def clean_injection(x, edge_index, num_inject, device):
    """
    清洁注入（零特征节点）
    """
    feat_dim = x.size(1)
    num_nodes = x.size(0)

    injected_info = []
    new_x = x.clone()
    new_edge_index = edge_index.clone()

    for i in range(num_inject):
        # 零特征
        injected_feat = torch.zeros(feat_dim, device=device)

        # 添加节点
        new_x = torch.cat([new_x, injected_feat.unsqueeze(0)], dim=0)
        new_node_idx = new_x.size(0) - 1

        # 随机选择邻居
        num_neighbors = np.random.randint(1, 3)
        neighbors = torch.randperm(num_nodes)[:num_neighbors].to(device)

        # 添加边
        new_edges = torch.stack([
            torch.cat([torch.tensor([new_node_idx], device=device).expand(num_neighbors), neighbors]),
            torch.cat([neighbors, torch.tensor([new_node_idx], device=device).expand(num_neighbors)])
        ], dim=0)
        new_edge_index = torch.cat([new_edge_index, new_edges], dim=1)

        injected_info.append({
            'node_idx': new_node_idx,
            'feature': injected_feat,
            'connected_nodes': neighbors.tolist()
        })

    return new_x, new_edge_index, injected_info


def run_defense(args, x, edge_index, labels, train_mask, val_mask, test_mask, device):
    """
    运行防御

    Args:
        args: 命令行参数
        x: 节点特征
        edge_index: 边索引
        labels: 节点标签
        train_mask: 训练集掩码
        val_mask: 验证集掩码
        test_mask: 测试集掩码
        device: 计算设备

    Returns:
        results: 防御结果
    """
    print(f"\n{'='*50}")
    print(f"Running {args.defense} defense...")
    print(f"{'='*50}")

    start_time = time.time()

    if args.defense == 'DAEG':
        # 使用 DAEG 防御
        purified_x, purified_edge_index, removed_nodes, detector = run_daeg_defense(
            model=None,
            x=x,
            edge_index=edge_index,
            labels=labels,
            labeled_mask=train_mask,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=device,
            hidden_dim=args.hidden_dim,
            alpha=args.alpha,
            beta=args.beta,
            tau_r=args.tau_r,
            tau_s=args.tau_s
        )

        # 在净化后的图上评估
        with torch.no_grad():
            logits, _ = detector(x, edge_index, purify=True)
            preds = logits.argmax(dim=-1)

        defense_time = time.time() - start_time
        print(f"Defense completed in {defense_time:.2f} seconds")
        print(f"Removed {len(removed_nodes)} suspicious nodes")

        results = {
            'predictions': preds,
            'removed_nodes': removed_nodes,
            'purified_x': purified_x,
            'purified_edge_index': purified_edge_index,
            'defense_time': defense_time
        }

    else:
        # 其他防御方法（使用标准 GNN 模型）
        from model import RGCNDetector, BotGCN

        if args.defense == 'BotRGCN':
            model = RGCNDetector(
                linear_channels=args.hidden_dim,
                out_channel=args.hidden_dim
            ).to(device)
        elif args.defense == 'BotGCN':
            model = BotGCN(
                hidden_dim=args.hidden_dim
            ).to(device)
        else:
            raise ValueError(f"Unknown defense method: {args.defense}")

        # 训练模型
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        early_stopping = EarlyStopping(patience=args.patience)

        best_val_acc = 0
        best_model_state = None

        for epoch in range(args.num_epochs):
            model.train()
            optimizer.zero_grad()

            # 前向传播（需要根据模型调整输入）
            # 这里简化处理
            logits = model(x[:, :768], x[:, 768:1536], x[:, 1536:1541], x[:, 1541:], edge_index, None)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])

            loss.backward()
            optimizer.step()

            # 验证
            model.eval()
            with torch.no_grad():
                logits = model(x[:, :768], x[:, 768:1536], x[:, 1536:1541], x[:, 1541:], edge_index, None)
                val_preds = logits[val_mask].argmax(dim=-1)
                val_acc = (val_preds == labels[val_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            if early_stopping(val_acc):
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

        # 加载最佳模型
        model.load_state_dict(best_model_state)

        # 测试
        model.eval()
        with torch.no_grad():
            logits = model(x[:, :768], x[:, 768:1536], x[:, 1536:1541], x[:, 1541:], edge_index, None)
            preds = logits.argmax(dim=-1)

        defense_time = time.time() - start_time
        print(f"Defense completed in {defense_time:.2f} seconds")

        results = {
            'predictions': preds,
            'removed_nodes': torch.tensor([], dtype=torch.long),
            'defense_time': defense_time
        }

    return results


def evaluate_attack_defense(args, original_x, original_edge_index, labels,
                            train_mask, val_mask, test_mask, device):
    """
    评估攻击和防御效果

    Args:
        args: 命令行参数
        original_x: 原始特征
        original_edge_index: 原始边索引
        labels: 标签
        train_mask, val_mask, test_mask: 数据划分
        device: 计算设备

    Returns:
        results: 完整的评估结果
    """
    results = {}

    # 1. 在原始图上训练一个检测模型
    print("\nTraining surrogate model on clean graph...")
    surrogate = SurrogateGCN(
        input_dim=original_x.size(1),
        hidden_dim=args.hidden_dim,
        output_dim=2
    ).to(device)

    optimizer = torch.optim.Adam(surrogate.parameters(), lr=args.lr)
    surrogate.train()

    for epoch in range(20):
        optimizer.zero_grad()
        out = surrogate(original_x, original_edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

    # 获取原始预测
    surrogate.eval()
    with torch.no_grad():
        original_out = surrogate(original_x, original_edge_index)
        original_preds = original_out.argmax(dim=-1)

    # 2. 选择目标节点（被正确分类的机器人节点）
    bot_mask = labels == 1
    correctly_classified_bots = (original_preds == labels) & bot_mask
    target_nodes = torch.where(correctly_classified_bots)[0]

    if len(target_nodes) == 0:
        print("Warning: No correctly classified bot nodes to target")
        target_nodes = torch.where(bot_mask)[0][:100]

    print(f"Number of target nodes: {len(target_nodes)}")

    # 3. 执行攻击
    attacked_x, attacked_edge_index, injected_info = run_attack(
        args, surrogate, original_x, original_edge_index,
        labels, target_nodes, device
    )

    # 记录注入节点索引
    injected_nodes = torch.tensor([info['node_idx'] for info in injected_info], device=device)
    num_original_nodes = original_x.size(0)

    # 4. 在攻击后的图上评估原始模型
    print("\nEvaluating surrogate on attacked graph...")
    surrogate.eval()
    with torch.no_grad():
        attacked_out = surrogate(attacked_x, attacked_edge_index)

        # 只评估原始节点
        attacked_out_original = attacked_out[:num_original_nodes]
        attacked_preds = attacked_out_original.argmax(dim=-1)

    # 计算攻击成功率
    asr = compute_attack_success_rate(
        target_nodes, original_preds, attacked_preds, labels
    )

    # 计算攻击后的准确率
    attacked_test_acc = (attacked_preds[test_mask] == labels[test_mask]).float().mean().item()

    print(f"Attack Success Rate (ASR): {asr:.4f}")
    print(f"Test Accuracy after attack: {attacked_test_acc:.4f}")

    results['attack'] = {
        'asr': asr,
        'test_acc_after_attack': attacked_test_acc,
        'num_injected': len(injected_info)
    }

    # 5. 执行防御
    defense_results = run_defense(
        args, attacked_x, attacked_edge_index, labels,
        train_mask, val_mask, test_mask, device
    )

    defense_preds = defense_results['predictions']
    removed_nodes = defense_results['removed_nodes']

    # 6. 计算防御指标
    # 测试集准确率（只在原始节点上计算）
    defense_test_preds = defense_preds[:num_original_nodes][test_mask]
    defense_test_labels = labels[test_mask]
    defense_test_acc = (defense_test_preds == defense_test_labels).float().mean().item()

    # 注入节点召回率 (IR)
    ir = compute_injected_node_recall(removed_nodes, injected_nodes)

    # 错误移除率 (FRR)
    frr = compute_false_removal_rate(removed_nodes, injected_nodes, attacked_x.size(0))

    # 计算 F1
    test_f1 = compute_evaluation_metrics(
        defense_test_labels.cpu().numpy(),
        defense_test_preds.cpu().numpy()
    )['f1_macro']

    print(f"\nDefense Results:")
    print(f"Test Accuracy: {defense_test_acc:.4f}")
    print(f"Test F1-macro: {test_f1:.4f}")
    print(f"Injected-node Recall (IR): {ir:.4f}")
    print(f"False Removal Rate (FRR): {frr:.4f}")

    results['defense'] = {
        'test_acc': defense_test_acc,
        'test_f1': test_f1,
        'ir': ir,
        'frr': frr,
        'num_removed': len(removed_nodes),
        'defense_time': defense_results.get('defense_time', 0)
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Graph Attack and Defense Experiments')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='twibot-22',
                        choices=['cresci-2015', 'cresci-2017', 'twibot-20', 'twibot-22'],
                        help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='./datasets',
                        help='Path to dataset')

    # 攻击参数
    parser.add_argument('--attack', type=str, default='IEA',
                        choices=['IEA', 'G-NIA', 'NIPA', 'LP-GIA', 'clean'],
                        help='Attack method')
    parser.add_argument('--injection_rate', type=float, default=0.05,
                        help='Rate of nodes to inject')
    parser.add_argument('--lambda_damp', type=float, default=0.01,
                        help='Damping coefficient for Hessian approximation')
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='Step size for text embedding optimization')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Similarity penalty coefficient')
    parser.add_argument('--budget', type=int, default=5,
                        help='Edge budget per injected node')
    parser.add_argument('--k_hop', type=int, default=2,
                        help='K-hop neighborhood for influence computation')
    parser.add_argument('--max_iterations', type=int, default=20,
                        help='Maximum iterations for IEA')

    # 防御参数
    parser.add_argument('--defense', type=str, default='DAEG',
                        choices=['DAEG', 'BotRGCN', 'BotGCN', 'GCN', 'R-GCN'],
                        help='Defense method')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Entropy weight in DAEG')
    parser.add_argument('--beta', type=float, default=0.4,
                        help='Smoothness weight in DAEG')
    parser.add_argument('--tau_r', type=float, default=1.0,
                        help='Suspiciousness threshold')
    parser.add_argument('--tau_s', type=float, default=0.35,
                        help='Similarity threshold')

    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs')
    parser.add_argument('--log', action='store_true',
                        help='Whether to log results')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 获取设备
    device = get_device()
    print(f"Using device: {device}")

    # 创建实验日志
    if args.log:
        logger = ExperimentLogger(
            log_path='./logs',
            exp_name=f'{args.dataset}_{args.attack}_{args.defense}'
        )

    # 运行多次实验
    all_results = []

    for run in range(args.runs):
        print(f"\n{'#'*60}")
        print(f"Run {run+1}/{args.runs}")
        print(f"{'#'*60}")

        # 设置当前运行的随机种子
        set_seed(args.seed + run)

        # 加载或生成模拟数据（实际使用时需要替换为真实数据加载）
        print(f"\nLoading dataset: {args.dataset}")

        # 这里使用模拟数据进行演示
        # 实际使用时需要替换为真实数据加载逻辑
        num_nodes = 1000
        feat_dim = 768 + 768 + 5 + 1  # des + tweet + num_prop + cat_prop

        x = torch.randn(num_nodes, feat_dim)
        num_edges = num_nodes * 10
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # 生成标签（0: human, 1: bot）
        labels = torch.randint(0, 2, (num_nodes,))

        # 划分数据集
        train_mask, val_mask, test_mask = split_dataset(num_nodes)

        # 移动到设备
        x = x.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

        # 评估攻击和防御
        results = evaluate_attack_defense(
            args, x, edge_index, labels,
            train_mask, val_mask, test_mask, device
        )

        all_results.append(results)

        if args.log:
            logger.log(run, results)

    # 计算平均结果
    print(f"\n{'#'*60}")
    print("Summary")
    print(f"{'#'*60}")

    avg_asr = np.mean([r['attack']['asr'] for r in all_results])
    avg_test_acc = np.mean([r['defense']['test_acc'] for r in all_results])
    avg_f1 = np.mean([r['defense']['test_f1'] for r in all_results])
    avg_ir = np.mean([r['defense']['ir'] for r in all_results])
    avg_frr = np.mean([r['defense']['frr'] for r in all_results])

    print(f"Average Attack Success Rate: {avg_asr:.4f}")
    print(f"Average Defense Test Accuracy: {avg_test_acc:.4f}")
    print(f"Average Defense F1-macro: {avg_f1:.4f}")
    print(f"Average Injected-node Recall: {avg_ir:.4f}")
    print(f"Average False Removal Rate: {avg_frr:.4f}")

    if args.log:
        logger.log('summary', {
            'avg_asr': avg_asr,
            'avg_test_acc': avg_test_acc,
            'avg_f1': avg_f1,
            'avg_ir': avg_ir,
            'avg_frr': avg_frr
        })
        logger.save()


if __name__ == '__main__':
    main()
