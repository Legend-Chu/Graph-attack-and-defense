import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def influence_function(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, labels):
    # 假设特征维度是768
    input_dim = des_tensor.size(1)
    hidden_dim = 128

    # 构建图数据
    x = torch.cat((des_tensor, tweets_tensor, num_prop, category_prop), dim=1)
    edge_index = edge_index.to(x.device)

    # 初始化模型
    model = GCN(input_dim, hidden_dim).to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[labels == 1], labels[labels == 1])
        loss.backward()
        optimizer.step()

    return out


def remove_injected_nodes(out, labels, threshold=0.5):
    # 计算节点的相似度
    sim_matrix = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1)

    # 找到标签为1的节点的索引
    bot_indices = torch.where(labels == 1)[0]
    injected_indices = []

    for idx in bot_indices:
        similar_nodes = torch.where(sim_matrix[idx] > threshold)[0]
        injected_indices.extend(similar_nodes.tolist())

    # 剔除重复的节点
    injected_indices = list(set(injected_indices))

    # 删除少量节点
    if len(injected_indices) > 1:
        # 保留特征差异大的节点
        injected_indices = sorted(injected_indices)[:len(injected_indices) // 2]

    return injected_indices


# 使用示例
edge_index = ...  # 你的边索引
des_tensor = ...  # 描述特征
tweets_tensor = ...  # 推文特征
num_prop = ...  # 数值特征
category_prop = ...  # 类别特征
labels = ...  # 标签

out = influence_function(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, labels)
injected_indices = remove_injected_nodes(out, labels)

# 更新特征，移除注入节点
new_des_tensor = des_tensor[~torch.tensor(injected_indices)]
new_tweets_tensor = tweets_tensor[~torch.tensor(injected_indices)]
new_num_prop = num_prop[~torch.tensor(injected_indices)]
new_category_prop = category_prop[~torch.tensor(injected_indices)]
new_labels = labels[~torch.tensor(injected_indices)]

# 保存新的数据
torch.save(new_des_tensor, 'new_des_tensor.pt')
torch.save(new_tweets_tensor, 'new_tweets_tensor.pt')
torch.save(new_num_prop, 'new_num_prop.pt')
torch.save(new_category_prop, 'new_category_prop.pt')
torch.save(new_labels, 'new_labels.pt')
