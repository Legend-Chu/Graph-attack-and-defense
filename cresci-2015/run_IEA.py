"""
run_IEA.py - 在 Cresci-2015 数据集上运行 IEA 攻击

使用方法:
    python run_IEA.py --num_inject 50 --budget 5
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IEA import IEA, SurrogateGCN
from utils import set_seed, get_device, split_dataset, compute_evaluation_metrics


def load_cresci2015_data(data_path='./'):
    """
    加载 Cresci-2015 数据集

    Returns:
        data: 包含特征、边、标签等信息的字典
    """
    # 尝试加载预处理数据
    data_file = os.path.join(data_path, 'processed_data.pt')
    if os.path.exists(data_file):
        return torch.load(data_file)

    # 否则返回 None
    return None


def generate_synthetic_data(num_nodes=5301, num_edges=20000, device='cuda'):
    """
    生成合成数据用于测试

    Cresci-2015 数据集统计:
    - Human: 1,950
    - Bot: 3,351
    - Total: 5,301
    """
    set_seed(42)

    # 特征维度 (des:768 + tweet:768 + num_prop:5 + cat_prop:3)
    feat_dim = 768 + 768 + 5 + 3

    # 生成特征
    des_tensor = torch.randn(num_nodes, 768)
    tweets_tensor = torch.randn(num_nodes, 768)
    num_prop = torch.randn(num_nodes, 5)
    category_prop = torch.randn(num_nodes, 3)

    # 合并特征
    x = torch.cat((des_tensor, tweets_tensor, num_prop, category_prop), dim=1)

    # 生成边
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # 生成标签 (0: human, 1: bot)
    # 按照 Cresci-2015 的比例
    num_humans = 1950
    num_bots = 3351
    labels = torch.cat([
        torch.zeros(num_humans, dtype=torch.long),
        torch.ones(num_bots, dtype=torch.long)
    ])
    # 打乱
    perm = torch.randperm(num_nodes)
    labels = labels[perm]

    return {
        'x': x.to(device),
        'edge_index': edge_index.to(device),
        'labels': labels.to(device),
        'des_tensor': des_tensor,
        'tweets_tensor': tweets_tensor,
        'num_prop': num_prop,
        'category_prop': category_prop
    }


def run_iea_on_cresci2015(args):
    """
    在 Cresci-2015 数据集上运行 IEA 攻击
    """
    device = get_device()
    print(f"Using device: {device}")

    # 加载数据
    print("\nLoading Cresci-2015 dataset...")
    data = load_cresci2015_data(args.data_path)

    if data is None:
        print("Processed data not found, using synthetic data for demonstration...")
        data = generate_synthetic_data(device=device)

    x = data['x']
    edge_index = data['edge_index']
    labels = data['labels']
    num_nodes = x.size(0)

    # 划分数据集
    train_mask, val_mask, test_mask = split_dataset(num_nodes)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    print(f"Dataset statistics:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.size(1)}")
    print(f"  Features: {x.size(1)}")
    print(f"  Bots: {(labels == 1).sum().item()}")
    print(f"  Humans: {(labels == 0).sum().item()}")

    # 训练代理模型
    print("\nTraining surrogate GCN model...")
    surrogate = SurrogateGCN(
        input_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        output_dim=2
    ).to(device)

    optimizer = torch.optim.Adam(surrogate.parameters(), lr=args.lr)

    surrogate.train()
    for epoch in range(args.surrogate_epochs):
        optimizer.zero_grad()
        out = surrogate(x, edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                val_preds = surrogate(x, edge_index)[val_mask].argmax(dim=-1)
                val_acc = (val_preds == labels[val_mask]).float().mean().item()
            print(f"Epoch {epoch+1}/{args.surrogate_epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

    # 获取原始预测
    surrogate.eval()
    with torch.no_grad():
        original_out = surrogate(x, edge_index)
        original_preds = original_out.argmax(dim=-1)

    original_test_acc = (original_preds[test_mask] == labels[test_mask]).float().mean().item()
    print(f"\nOriginal test accuracy: {original_test_acc:.4f}")

    # 选择目标节点（被正确分类的机器人节点）
    bot_mask = labels == 1
    correctly_classified_bots = (original_preds == labels) & bot_mask
    target_nodes = torch.where(correctly_classified_bots)[0]

    if len(target_nodes) == 0:
        print("Warning: No correctly classified bot nodes. Using all bot nodes as targets.")
        target_nodes = torch.where(bot_mask)[0]

    print(f"Number of target nodes: {len(target_nodes)}")

    # 创建 IEA 攻击器
    print("\nInitializing IEA attacker...")
    iea = IEA(
        surrogate_model=surrogate,
        feat_dim=x.size(1),
        num_classes=2,
        lambda_damp=args.lambda_damp,
        num_hessian_steps=args.num_hessian_steps,
        step_size=args.step_size,
        gamma=args.gamma,
        budget=args.budget,
        k_hop=args.k_hop,
        top_n_candidates=args.top_n_candidates,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        device=device
    )

    # 执行攻击
    num_inject = int(num_nodes * args.injection_rate)
    print(f"\nRunning IEA attack with {num_inject} injected nodes...")

    import time
    start_time = time.time()

    attacked_x, attacked_edge_index, injected_info = iea(
        x, edge_index, target_nodes, labels, num_inject
    )

    attack_time = time.time() - start_time
    print(f"Attack completed in {attack_time:.2f} seconds")

    # 在攻击后的图上评估
    print("\nEvaluating attack effectiveness...")
    with torch.no_grad():
        attacked_out = surrogate(attacked_x, attacked_edge_index)

        # 只评估原始节点
        attacked_out_original = attacked_out[:num_nodes]
        attacked_preds = attacked_out_original.argmax(dim=-1)

    # 计算指标
    attacked_test_acc = (attacked_preds[test_mask] == labels[test_mask]).float().mean().item()

    # 攻击成功率
    target_attacked = (attacked_preds[target_nodes] != labels[target_nodes]).sum().item()
    asr = target_attacked / len(target_nodes) if len(target_nodes) > 0 else 0

    # 注入节点检测率
    injected_indices = [info['node_idx'] for info in injected_info]
    injected_preds = attacked_out[injected_indices].argmax(dim=-1)
    detection_rate = (injected_preds == 1).float().mean().item()  # 被检测为 bot 的比例

    print(f"\n{'='*50}")
    print("Attack Results:")
    print(f"{'='*50}")
    print(f"Original test accuracy: {original_test_acc:.4f}")
    print(f"Attacked test accuracy: {attacked_test_acc:.4f}")
    print(f"Accuracy drop: {original_test_acc - attacked_test_acc:.4f}")
    print(f"Attack Success Rate (ASR): {asr:.4f}")
    print(f"Injected nodes detected as bots: {detection_rate:.4f}")
    print(f"Number of injected nodes: {len(injected_info)}")
    print(f"Attack time: {attack_time:.2f}s")

    # 保存结果
    if args.save_results:
        results = {
            'attacked_x': attacked_x.cpu(),
            'attacked_edge_index': attacked_edge_index.cpu(),
            'injected_info': injected_info,
            'original_test_acc': original_test_acc,
            'attacked_test_acc': attacked_test_acc,
            'asr': asr,
            'detection_rate': detection_rate,
            'attack_time': attack_time
        }
        torch.save(results, 'iea_attack_results.pt')
        print(f"\nResults saved to iea_attack_results.pt")

    return {
        'attacked_x': attacked_x,
        'attacked_edge_index': attacked_edge_index,
        'injected_info': injected_info,
        'asr': asr,
        'attacked_test_acc': attacked_test_acc
    }


def main():
    parser = argparse.ArgumentParser(description='IEA Attack on Cresci-2015')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='./',
                        help='Path to dataset')

    # 攻击参数
    parser.add_argument('--injection_rate', type=float, default=0.05,
                        help='Rate of nodes to inject')
    parser.add_argument('--budget', type=int, default=5,
                        help='Edge budget per injected node')
    parser.add_argument('--lambda_damp', type=float, default=0.01,
                        help='Damping coefficient for Hessian')
    parser.add_argument('--num_hessian_steps', type=int, default=10,
                        help='Number of Hessian-vector product steps')
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='Step size for embedding optimization')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Similarity penalty coefficient')
    parser.add_argument('--k_hop', type=int, default=2,
                        help='K-hop neighborhood size')
    parser.add_argument('--top_n_candidates', type=int, default=128,
                        help='Number of candidate neighbors')
    parser.add_argument('--max_iterations', type=int, default=20,
                        help='Maximum optimization iterations')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Convergence tolerance')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension of surrogate model')
    parser.add_argument('--surrogate_epochs', type=int, default=50,
                        help='Number of epochs to train surrogate')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_results', action='store_true',
                        help='Whether to save attack results')

    args = parser.parse_args()

    set_seed(args.seed)
    run_iea_on_cresci2015(args)


if __name__ == '__main__':
    main()
