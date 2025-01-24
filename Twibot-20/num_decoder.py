from model import oriRGCN, num_decoder, num_encoder, encoder
from Dataset import Twibot20
import torch
from utils import init_weights
import torch.nn.functional as F
import numpy as np

device = 'cuda:0'
lr, weight_decay=1e-5, 5e-2

root='./processed_data/'
# 加载数据集
dataset=Twibot20(root=root, device=device)
# 从数据集中获取各种张量数据，包括描述向量、推文特征、数值属性、分类属性、边索引、边类型、标签以及训练/验证/测试索引
des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx = dataset.dataloader()

# 初始化模型
model = num_decoder().to(device) # 数值解码器模型
num_encoder_model = num_encoder().to(device) # 数值编码器模型
ori_model = oriRGCN().to(device) # 原始图神经网络模型
# 加载原始模型的预训练权重
ori_model.load_state_dict(torch.load('model/model_ori_rgcn.pth'))

# 从原始模型中提取数值解码器的权重和偏置
model_dict = ori_model.state_dict()
num_weight = model_dict['linear_relu_num_prop.0.weight'].clone().detach()
num_bias = model_dict['linear_relu_num_prop.0.bias'].clone().detach()
num_encoder_dict = {'linear_relu_num_prop.0.weight': num_weight, 'linear_relu_num_prop.0.bias': num_bias}
num_encoder_model.load_state_dict(num_encoder_dict)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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

num_dict = np.load('processed_data/num_properties_meanstd.npy', allow_pickle=True).item()
mean = torch.tensor([num_dict['followers_count_mean'], num_dict['active_days_mean'], num_dict['screen_name_length_mean'], num_dict['following_count_mean'], num_dict['statues_mean']], device=device)
std = torch.tensor([num_dict['followers_count_std'], num_dict['active_days_std'], num_dict['screen_name_length_std'], num_dict['following_count_std'], num_dict['statues_std']], device=device)

features = encoder_model(des_tensor,tweets_tensor,num_prop,category_prop)
num_features = features[:, 16:24]
num_features = torch.tensor(num_features, device=device)

loss = []
def train(epoch):
    model.train()
    output = model(num_features)
    # output = F.softmax(output, dim=1)
    loss_train1 = torch.sum(abs(output - num_prop))
    output_features = num_encoder_model(output)
    loss_train2 = 0.01 * torch.sum(abs(output_features-num_features))
    loss_train = loss_train1 + loss_train2
    loss.append(loss_train)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()))
    return loss_train

def rmse(predictions, targets):
    mse = F.mse_loss(predictions, targets)
    rmse = torch.sqrt(mse)
    return rmse

def test():
    model.eval()
    output = model(num_features)
    loss_test1 = torch.sum(abs(output - num_prop))
    output_features = num_encoder_model(output)
    loss_test2 = 0.01 * torch.sum(abs(output_features-num_features))
    loss_test = loss_test1 + loss_test2
    print("test_loss=", loss_test)
    sum_rmse = 0
    for i in range(output.shape[0]):
        rmse_i = rmse(num_prop[i], output[i])
        sum_rmse = rmse_i + sum_rmse
    print("average_rmse:", sum_rmse / output.shape[0])

model.apply(init_weights)
# model.load_state_dict(torch.load('model/model_num_decoder.pth'))

epochs = 10000
for epoch in range(epochs):
    train(epoch)

torch.save(model.state_dict(), 'model/model_num_decoder.pth')

test()