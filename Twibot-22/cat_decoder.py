from model import oriRGCN, cat_decoder, encoder
from Dataset import Twibot22
import torch
from torch import nn
from utils import bot_accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData
import torch.nn.functional as F

import pandas as pd
import numpy as np

device = 'cuda:0'
lr,weight_decay=1e-5,5e-2

root='./processed_data/'

dataset=Twibot22(root=root,device=device,process=False,save=False)
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()

model = cat_decoder().to(device)

ori_model = oriRGCN().to(device)
ori_model.load_state_dict(torch.load('model/model_ori_rgcn.pth'))
model_dict = ori_model.state_dict()

optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)



encoder_model = encoder().to(device)
model_dict = ori_model.state_dict()
des_weight = model_dict['linear_relu_des.0.weight'].clone().detach()
des_bias = model_dict['linear_relu_des.0.bias'].clone().detach()
tweet_weight = model_dict['linear_relu_tweet.0.weight'].clone().detach()
tweet_bias = model_dict['linear_relu_tweet.0.bias'].clone().detach()
num_weight = model_dict['linear_relu_num_prop.0.weight'].clone().detach()
num_bias = model_dict['linear_relu_num_prop.0.bias'].clone().detach()
cat_weight = model_dict['linear_relu_cat_prop.0.weight'].clone().detach()
cat_bias = model_dict['linear_relu_cat_prop.0.bias'].clone().detach()
encoder_dict = {'linear_relu_des.0.weight': des_weight, 'linear_relu_des.0.bias': des_bias,
                    'linear_relu_tweet.0.weight': tweet_weight, 'linear_relu_tweet.0.bias': tweet_bias,
                    'linear_relu_num_prop.0.weight': num_weight, 'linear_relu_num_prop.0.bias': num_bias,
                    'linear_relu_cat_prop.0.weight': cat_weight, 'linear_relu_cat_prop.0.bias': cat_bias
                    }
encoder_model.load_state_dict(encoder_dict)


features = encoder_model(des_tensor,tweets_tensor,num_prop,category_prop)
cat_features = features[:, 24:]
cat_features = torch.tensor(cat_features, device=device)


def train(epoch):
    model.train()
    output = model(cat_features)
    loss_train = torch.sum(abs(output-category_prop))
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()))
    # print(output)
    return loss_train


def rmse(predictions, targets):
    mse = F.mse_loss(predictions, targets)
    rmse = torch.sqrt(mse)
    return rmse


def test():
    model.eval()
    output = model(cat_features)
    loss_test = torch.sum(abs(output-category_prop))
    print("test_loss=", loss_test)
    print("rmse:", rmse(category_prop, output))


model.apply(init_weights)
# model.load_state_dict(torch.load('model/model_cat_decoder.pth'))
#
epochs = 10000
for epoch in range(epochs):
    train(epoch)


torch.save(model.state_dict(), 'model/model_cat_decoder.pth')

test()