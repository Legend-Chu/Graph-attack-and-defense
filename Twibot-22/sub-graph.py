from Dataset import Twibot22
import torch
from torch_geometric.utils import subgraph
import numpy as np

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

root = './processed_data/'
path = './sub_processed_data/'

dataset = Twibot22(root=root, device='cpu', process=False, save=False)
des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx = dataset.dataloader()

num_nodes = torch.max(edge_index).int()
start_size = 10000
center_nodes = torch.randperm(num_nodes)[:start_size]
print('center_nodes:', center_nodes)
subgraph_size = 50000
subgraph_nodes = set(center_nodes.tolist())
edge_indices = set()
visited_nodes = set(center_nodes.tolist())
queue = list(center_nodes.tolist())
while len(queue) > 0:
    node = queue.pop(0)
    print("Processing node:", node)
    out_neighbors = edge_index[1][edge_index[0] == node]
    print("Out-neighbors of node:", node, "are", out_neighbors.tolist())
    for neighbor in out_neighbors:
        if neighbor not in subgraph_nodes:
            subgraph_nodes.add(neighbor)
            visited_nodes.add(neighbor)
            edge_indices.add(torch.tensor([node, neighbor]))
            queue.append(neighbor)
    in_neighbors = edge_index[0][edge_index[1] == node]
    print("In-neighbors of node:", node, "are", in_neighbors.tolist())
    for neighbor in in_neighbors:
        if neighbor not in subgraph_nodes:
            subgraph_nodes.add(neighbor)
            visited_nodes.add(neighbor)
            edge_indices.add(torch.tensor([node, neighbor]))
            queue.append(neighbor)

    subgraph_nodes_count = torch.tensor(list(set(subgraph_nodes)))
    subgraph_nodes_count = torch.unique(subgraph_nodes_count)
    print(torch.sort(subgraph_nodes_count))
    if len(subgraph_nodes_count) >= subgraph_size:
        break

subgraph_nodes = torch.tensor(list(set(subgraph_nodes)))
subgraph_nodes = torch.unique(subgraph_nodes)
sorted = torch.sort(subgraph_nodes)
print('subgraph_nodes', len(subgraph_nodes), sorted)

edge_indices = list(edge_indices)
if len(edge_indices) > 0:
    edge_indices = torch.stack(edge_indices).t()
else:
    print("No edges found in subgraph.")

sub_edge_index, mapping = subgraph(subgraph_nodes, edge_index)

print(sub_edge_index)
print(sub_edge_index.shape)

train_mask = torch.eq(train_idx.view(-1, 1), subgraph_nodes.view(1, -1))
train_indices = torch.where(train_mask)
sub_train_idx = subgraph_nodes[train_indices[1]]

val_mask = torch.eq(val_idx.view(-1, 1), subgraph_nodes.view(1, -1))
val_indices = torch.where(val_mask)
sub_val_idx = subgraph_nodes[val_indices[1]]

test_mask = torch.eq(test_idx.view(-1, 1), subgraph_nodes.view(1, -1))
test_indices = torch.where(test_mask)
sub_test_idx = subgraph_nodes[test_indices[1]]

torch.save(sub_train_idx, path + 'train_idx.pt')
torch.save(sub_val_idx, path + 'val_idx.pt')
torch.save(sub_test_idx, path + 'test_idx.pt')

torch.save(sub_edge_index, path + 'edge_index.pt')

mask = torch.isin(edge_index, sub_edge_index)
indices = torch.nonzero(mask.all(dim=0)).squeeze()
sub_edge_type = edge_type[indices]

torch.save(sub_edge_type, path + 'edge_type.pt')

torch.save(subgraph_nodes, path + 'subgraph_nodes.pt')
