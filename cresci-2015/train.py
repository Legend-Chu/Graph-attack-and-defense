from model import BotRGCN, oriRGCN, HGTDetector, SHGNDetector,RGCNDetector,BotGCN
from DatasetCresci import Twibot22
import torch
from torch import nn
from utils import bot_accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData

import pandas as pd
import numpy as np

device = 'cuda:0'
embedding_size, dropout, lr, weight_decay = 32, 0.1, 1e-3, 5e-2

root='./processed_data/'

dataset = Twibot22(root='./processed_data/',device=device)
(des_tensor,tweet_channel, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx) = dataset.dataloader()
# train RGCN submitted model
# model = RGCNDetector().to(device)
# model = SHGNDetector().to(device)
# model = oriRGCN().to(device)
# model = BotGCN().to(device)
model = HGTDetector().to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def train(epoch):
    # print(edge_type,edge_index)
    output = model(des_tensor, tweet_channel, num_prop, category_prop, edge_index, edge_type)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = bot_accuracy(output[train_idx], labels[train_idx])
    acc_val = bot_accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()), )
    return acc_train, loss_train

def test():
    model.eval()
    output = model(des_tensor, tweet_channel, num_prop, category_prop, edge_index, edge_type)
    loss_test = loss(output[test_idx], labels[test_idx])
    acc_test = bot_accuracy(output[test_idx], labels[test_idx])
    output=output.max(1)[1].to('cpu').detach().numpy()
    label=labels.to('cpu').detach().numpy()
    f1=f1_score(label[test_idx],output[test_idx])
    precision=precision_score(label[test_idx],output[test_idx])
    recall=recall_score(label[test_idx],output[test_idx])

    fpr, tpr, thresholds = roc_curve(label[test_idx], output[test_idx], pos_label=1)
    Auc=auc(fpr, tpr)
    print("test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            "precision= {:.4f}".format(precision.item()),
            "recall= {:.4f}".format(recall.item()),
            "f1_score= {:.4f}".format(f1.item()),
            #"mcc= {:.4f}".format(mcc.item()),
            "auc= {:.4f}".format(Auc.item()),
            )


model.apply(init_weights)
# model.load_state_dict(torch.load('model/model_HGT.pth'))

epochs = 100
for epoch in range(epochs):
    train(epoch)

# torch.save(model.state_dict(), 'model/model_ori_rgcn.pth')
# torch.save(model.state_dict(),'model/model_SimpleHGN.pth')
# torch.save(model.state_dict(),'model/model_RGCN.pth')
# torch.save(model.state_dict('model/model_GCN.pth'))
torch.save(model.state_dict(),'model/model_HGT.pth')

test()