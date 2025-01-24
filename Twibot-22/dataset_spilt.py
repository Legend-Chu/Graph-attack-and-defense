from subDataset import Twibot22
import torch
from torch_geometric.utils import subgraph
import numpy as np

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)
device = 'cpu'
root = './processed_data/'
path = './sub_processed_data/'

subgraph_nodes = torch.load(path + "subgraph_nodes.pt", map_location='cpu').values.tolist()
subgraph_nodes = torch.tensor(subgraph_nodes).to(device)
labels = torch.load(root + "label.pt").to(device)
edge_index = torch.load(path + "edge_index.pt", map_location='cpu').to(device)
labels = labels[subgraph_nodes]
new_ids = torch.arange(len(subgraph_nodes))
edge_index[0,:] = torch.tensor([new_ids[subgraph_nodes == i] for i in edge_index[0,:]])
edge_index[1,:] = torch.tensor([new_ids[subgraph_nodes == i] for i in edge_index[1,:]])
print('load finished')

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

num_nodes = len(subgraph_nodes)
train_size = int(num_nodes * train_ratio)
val_size = int(num_nodes * val_ratio)
test_size = num_nodes - train_size - val_size

shuffled_indices = torch.randperm(num_nodes)

sub_train_idx = shuffled_indices[:train_size]
sub_val_idx = shuffled_indices[train_size:train_size+val_size]
sub_test_idx = shuffled_indices[train_size+val_size:]
print('split finished')

torch.save(sub_train_idx, path + 'train_idx.pt')
torch.save(sub_val_idx, path + 'val_idx.pt')
torch.save(sub_test_idx, path + 'test_idx.pt')


print('train,val,test',len(sub_train_idx),len(sub_val_idx),len(sub_test_idx))

train_mask_human = []
train_mask_bot = []
for i in sub_train_idx:
    if labels[i] == 0:
        train_mask_human.append(i)
    else:
        train_mask_bot.append(i)

val_mask_human = []
val_mask_bot = []
for i in sub_val_idx:
    if labels[i] == 0:
        val_mask_human.append(i)
    else:
        val_mask_bot.append(i)

test_mask_human = []
test_mask_bot = []
for i in sub_test_idx:
    if labels[i] == 0:
        test_mask_human.append(i)
    else:
        test_mask_bot.append(i)

human_mask = np.append(train_mask_human, val_mask_human, axis=0)
human_mask = np.append(human_mask, test_mask_human, axis=0)
bot_mask = np.append(train_mask_bot, val_mask_bot, axis=0)
bot_mask = np.append(bot_mask, test_mask_bot, axis=0)

train_mask_human = torch.tensor(train_mask_human)
train_mask_bot = torch.tensor(train_mask_bot)
val_mask_human = torch.tensor(val_mask_human)
val_mask_bot = torch.tensor(val_mask_bot)
test_mask_human = torch.tensor(test_mask_human)
test_mask_bot = torch.tensor(test_mask_bot)
human_mask = torch.tensor(human_mask)
bot_mask = torch.tensor(bot_mask)

print('train human:', train_mask_human.shape)
print('train bot:', train_mask_bot.shape)
print('val human:', val_mask_human.shape)
print('val bot:', val_mask_bot.shape)
print('test human:', test_mask_human.shape)
print('test bot:', test_mask_bot.shape)


bot_mask = bot_mask.to(device)


degree_tensor = torch.zeros((len(labels), ), dtype=torch.long, device=device)
degree_tensor.index_add_(0, edge_index[0], torch.ones((edge_index.size(1), ), dtype=torch.long, device=device))
bot_degree = degree_tensor[bot_mask].float().mean().item()

print('Robot average degree:', bot_degree)

human_count = len(human_mask)
robot_count = len(bot_mask)

print('Robot average degree:', bot_degree)
print('Number of human users:', human_count)
print('Number of robot users:', robot_count)
