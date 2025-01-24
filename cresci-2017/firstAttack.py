import os
import math
import argparse
import numpy as np
import torch.utils.data as Data
np_load_old = np.load
np.aload = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
from gnia import GNIA
from utils import *
from model import oriRGCN, encoder, RGCN, num_decoder, cat_decoder, RGCN_weight
from DatasetCresci import Twibot22
import torch
from torch import nn
import tqdm
from utils import accuracy
import numpy as np
import matplotlib.pyplot as plt

def main(opts):
    # hyperparameters
    gpu_id = opts['gpu']
    seed = opts['seed']
    connect = opts['connect']
    multi = opts['multiedge']
    discrete = opts['discrete']
    suffix = opts['suffix']
    attr_tau = float(opts['attrtau']) if opts['attrtau'] != None else opts['attrtau']
    edge_tau = float(opts['edgetau']) if opts['edgetau'] != None else opts['edgetau']
    lr = opts['lr']
    patience = opts['patience']
    best_score = opts['best_score']
    counter = opts['counter']
    nepochs = opts['nepochs']
    st_epoch = opts['st_epoch']
    epsilon_start = opts['epsst']
    epsilon_end = 0
    epsilon_decay = opts['epsdec']
    total_steps = 500
    batch_size = opts['batchsize']

    # Set up directories for saving checkpoints
    ckpt_save_dirs = 'checkpoint/bot_gnia/'
    model_save_file = ckpt_save_dirs
    if not os.path.exists(ckpt_save_dirs):
        os.makedirs(ckpt_save_dirs)

    # Set the GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing data
    root = './processed_data/'
    dataset = Twibot22(root=root, device=device)
    # train_mask, val_mask, test_mask是训练集 验证集和测试集的节点索引
    des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels_np, train_mask, val_mask, test_mask = dataset.dataloader()

    # Separate train, validation, and test masks into human and bot masks
    train_mask_human = []
    train_mask_bot = []
    # 训练集标签
    for i in train_mask:
        if labels_np[i] == 0:
            train_mask_human.append(i)
        else:
            train_mask_bot.append(i)

    val_mask_human = []
    val_mask_bot = []
    # 验证集标签
    for i in val_mask:
        if labels_np[i] == 0:
            val_mask_human.append(i)
        else:
            val_mask_bot.append(i)

    test_mask_human = []
    test_mask_bot = []
    # 测试集标签
    for i in test_mask:
        if labels_np[i] == 0:
            test_mask_human.append(i)
        else:
            test_mask_bot.append(i)

    # 按照机器人和人划分数据集
    human_mask = np.append(train_mask_human, val_mask_human, axis=0)
    human_mask = np.append(human_mask, test_mask_human, axis=0)
    bot_mask = np.append(train_mask_bot, val_mask_bot, axis=0)
    bot_mask = np.append(bot_mask, test_mask_bot, axis=0)

    # 按照三个集中的人和机器人划分
    train_mask_human = torch.tensor(train_mask_human)
    train_mask_bot = torch.tensor(train_mask_bot)
    val_mask_human = torch.tensor(val_mask_human)
    val_mask_bot = torch.tensor(val_mask_bot)
    test_mask_human = torch.tensor(test_mask_human)
    test_mask_bot = torch.tensor(test_mask_bot)
    human_mask = torch.tensor(human_mask)
    bot_mask = torch.tensor(bot_mask)

    ori_model = oriRGCN().to(device)
    ori_model.load_state_dict(torch.load('model/model_ori_rgcn.pth'))
    encoder_model = encoder().to(device)

    # Extract specific layer weights from original model to the encoder
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

    # Load decoder models for numerical and categorical properties
    num_decoder_model = num_decoder().to(device)
    num_decoder_model.load_state_dict(torch.load('model/model_num_decoder.pth'))
    cat_decoder_model = cat_decoder().to(device)
    cat_decoder_model.load_state_dict(torch.load('model/model_cat_decoder.pth'))

    # Load mean and standard deviation for numerical properties
    num_dict = np.load('processed_data/num_properties_meanstd.npy', allow_pickle=True).item()
    num_mean = torch.tensor(
        [num_dict['followers_count_mean'], num_dict['active_days_mean'], num_dict['screen_name_length_mean'],
         num_dict['following_count_mean'], num_dict['statues_mean']], device=device)
    num_std = torch.tensor(
        [num_dict['followers_count_std'], num_dict['active_days_std'], num_dict['screen_name_length_std'],
         num_dict['following_count_std'], num_dict['statues_std']], device=device)

    # 使用encoder模型生成特征
    features = encoder_model(des_tensor, tweets_tensor, num_prop, category_prop)
    features = torch.tensor(features, device='cpu')
    features = np.array(features)
    features = sp.csr_matrix(features, dtype=np.float32)

    # Load adjacency matrix and preprocess it
    adj = load_adj(edge_index)
    n = adj.shape[0]
    nc = labels_np.max() + 1
    nfeat = features.shape[1]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(n)
    adj[adj > 1] = 1

    if connect:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels_np = labels_np[lcc]
        n = adj.shape[0]
        print('Nodes num:', n)

    # Convert adjacency matrix to tensor and normalize it
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    nor_adj_tensor = normalize_tensor(adj_tensor)

    # Prepare features for model input
    feat = torch.from_numpy(features.todense().astype('double')).float().to(device)
    feat_max = feat.max(0).values
    feat_min = feat.min(0).values
    labels = labels_np
    degree = adj.sum(1)
    deg = torch.FloatTensor(degree).flatten().to(device)
    feat_num = int(features.sum(1).mean())

    # Calculate epsilon thresholds for each step
    eps_threshold = [epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps / epsilon_decay) for steps in range(total_steps)]

    # Load RGCN detection model and its weights
    detect_model = RGCN().to(device)
    rgcn_weight = model_dict['rgcn.weight'].clone().detach()
    rgcn_root = model_dict['rgcn.root'].clone().detach()
    RGCN_dict = {'rgcn.weight': rgcn_weight, 'rgcn.root': rgcn_root}
    detect_model.load_state_dict(RGCN_dict)

    # Disable gradient calculation for detection model
    detect_model.eval()
    for p in detect_model.parameters():
        p.requires_grad = False

    node_emb = detect_model(feat, edge_index, edge_type)

    # Load and prepare RGCN weight model
    RGCN_weight_model = RGCN_weight().to(device)
    W_weight = detect_model.rgcn.weight.data.detach()
    W1 = W_weight[0]
    W2 = W_weight[1]
    W = torch.cat((W1, W2), dim=0).t()
    W = RGCN_weight_model(W)

    # 对整个数据集进行模型推理
    logits = detect_model(feat, edge_index, edge_type)
    logp = F.log_softmax(logits, dim=1)  # 计算log softmax
    acc = accuracy(logp, labels)  # 计算总体准确率
    print('Acc:', acc)
    print('Train Acc:', accuracy(logp[train_mask], labels[train_mask]))
    print('Valid Acc:', accuracy(logp[val_mask], labels[val_mask]))
    print('Test Acc:', accuracy(logp[test_mask], labels[test_mask]))
    print('*' * 30)

    # Initialization
    model = GNIA(labels, nfeat, W, discrete, device, feat_min=feat_min, feat_max=feat_max, feat_num=feat_num,
                 attr_tau=attr_tau, edge_tau=edge_tau).to(device)
    stopper = EarlyStop_loss(patience=patience)

    # 根据用户选择初始化优化器
    if opts['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
    elif opts['optimizer'] == 'RMSprop':
        optimizer = torch.optim.RMSprop([{'params': model.parameters()}], lr=lr, weight_decay=0)
    else:
        raise ValueError('Unsupported argument for the optimizer')

    # 准备训练数据
    x = torch.LongTensor(train_mask_bot)
    y = labels[train_mask_bot].to(torch.device('cpu'))
    torch_dataset = Data.TensorDataset(x, y)
    batch_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, num_workers=24)

    # 如果开始训练的轮数不为0，加载之前的模型状态
    if st_epoch != 0:
        model.load_state_dict(torch.load(model_save_file + 'checkpoint.pt'))
        stopper.best_score = best_score
        stopper.counter = counter

    # ********************Train Part********************
    for epoch in range(st_epoch, nepochs):
        training = True
        print("Epoch {:05d}".format(epoch))
        train_atk_success = []
        val_atk_success = []
        train_loss_arr = []
        eps = eps_threshold[epoch] if epoch < len(eps_threshold) else eps_threshold[-1]

        for batch_x, _ in batch_loader:
            loss_arr = []
            for train_batch in batch_x:
                try:
                    target = np.array([train_batch])
                    target_deg = int(sum([degree[i].item() for i in target]))
                    budget = int(min(round(target_deg / 2), round(degree.mean()))) if multi else 1
                    ori = labels_np[target].item()
                    best_wrong_label = 0
                    one_order_nei = adj[target].nonzero()[1]
                    # 计算归一化邻接矩阵
                    tar_norm_adj = nor_adj_tensor[target.item()].to_dense()
                    norm_a_target = tar_norm_adj[one_order_nei].unsqueeze(1)
                    # 调用G-NIA模型生成注入的特征和边信息
                    inj_feat, disc_score, masked_score_idx = model(target, one_order_nei, budget, feat, norm_a_target,
                                                               node_emb,
                                                               W[ori], W[best_wrong_label], train_flag=training,
                                                               eps=eps)
                    # 解码并组合G-NIA生成的新特征
                    new_num = num_decoder_model(inj_feat[16:24])
                    new_cat_float = cat_decoder_model(inj_feat[24:])
                    new_cat = torch.tensor(0 if torch.all(new_cat_float) <= 0.5 else 1, device = device).reshape(1, 1)
                    new_des_tensor = torch.cat((des_tensor, torch.zeros(1, 768).to(device)), 0)
                    new_tweets_tensor = torch.cat((tweets_tensor, torch.zeros(1, 768).to(device)), 0)
                    new_num_prop = torch.cat((num_prop, new_num.unsqueeze(0)), 0)
                    new_category_prop = torch.cat((category_prop, new_cat), 0)
                    new_edge_index = gen_extend_edge_index(edge_index, adj_tensor, disc_score, masked_score_idx, device)

                    extend_edge_type = torch.tensor([1]).to(device)
                    new_edge_type = torch.cat((edge_type, extend_edge_type), 0)
                    # 使用原始检测模型进行检测
                    new_logits = ori_model(new_des_tensor, new_tweets_tensor, new_num_prop, new_category_prop, new_edge_index, new_edge_type)  # (10005,41)
                except IndexError:
                    continue

                else:
                    loss = F.relu(new_logits[target, ori] - new_logits[target, best_wrong_label])
                    new_logp = F.log_softmax(new_logits, dim=1)
                    train_atk_success.append(int(0 == new_logits[target].argmax(1).item()))
                    loss_arr.append(loss)

            train_loss = sum(loss_arr)
            optimizer.zero_grad()
            train_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.1)
            optimizer.step()
            train_loss_arr.append((train_loss / len(loss_arr)).detach().cpu().item())

            del loss_arr, train_loss
        print('Training set Loss:', np.array(train_loss_arr).mean())
        print('Training set: Attack success rate:', np.array(train_atk_success).mean())
        # del train_loss_arr, train_atk_success
        torch.cuda.empty_cache()

    # ********************Valid Part********************
    val_loss_arr = []
    training = False
    for val_batch in val_mask_bot:
        target = np.array([val_batch])
        target_deg = int(sum([degree[i].item() for i in target]))
        budget = int(min(round(target_deg / 2), round(degree.mean()))) if multi else 1
        ori = labels_np[target].item()
        best_wrong_label = 0
        one_order_nei = adj[target].nonzero()[1]
        tar_norm_adj = nor_adj_tensor[target.item()].to_dense()
        norm_a_target = tar_norm_adj[one_order_nei].unsqueeze(1)
        inj_feat, disc_score, masked_score_idx = model(target, one_order_nei, budget, feat, norm_a_target, node_emb,
                                                           W[ori], W[best_wrong_label], train_flag=training, eps=eps)
        new_num = num_decoder_model(inj_feat[16:24])
        new_cat_float = cat_decoder_model(inj_feat[24:])
        new_cat = torch.tensor(0 if torch.all(new_cat_float) <= 0.5 else 1, device=device).reshape(1, 1)
        new_des_tensor = torch.cat((des_tensor, torch.zeros(1, 768).to(device)), 0)
        new_num_prop = torch.cat((num_prop, new_num.unsqueeze(0)), 0)
        new_category_prop = torch.cat((category_prop, new_cat), 0)
        new_edge_index = gen_extend_edge_index(edge_index, adj_tensor, disc_score, masked_score_idx, device)
        extend_edge_type = torch.tensor([1]).to(device)
        new_edge_type = torch.cat((edge_type, extend_edge_type), 0)
        # use detect model
        new_logits = ori_model(new_des_tensor, new_tweets_tensor, new_num_prop, new_category_prop,
                                   new_edge_index, new_edge_type)
        loss = F.relu(new_logits[target, ori] - new_logits[target, best_wrong_label])
        new_logp = F.log_softmax(new_logits, dim=1)
        val_atk_success.append(int(0 == new_logits[target].argmax(1).item()))
        val_loss_arr.append(loss.detach().cpu().item())
    print('Validation set Loss:', np.array(val_loss_arr).mean())
    print('Validation set: Attack success rate:', np.array(val_atk_success).mean())

    val_loss = np.array(val_loss_arr).mean()

    # if stopper.step(val_loss, model, model_save_file):
    #    break
    # del val_loss_arr, val_atk_success
    torch.cuda.empty_cache()

    # ********************Test Part********************
    names = locals()
    training = False
    # model.load_state_dict(torch.load(model_save_file + 'checkpoint.pt'))
    for p in model.parameters():
        p.requires_grad = False
    for dset in ['test']:
        names[dset + '_bot_atk'] = []
        names[dset + '_new_node'] = []

        # 后续test阶段直接在原先的tensor后面反复添加注入节点
        new_des_tensor = des_tensor
        new_tweets_tensor = tweets_tensor
        new_num_prop = num_prop
        new_category_prop = category_prop
        new_edge_index = edge_index
        new_edge_type = edge_type
        # 0是human 1是bot
        new_labels = labels_np

        bot_count = 0
        human_count = 0

        for batch in names[dset + '_mask_bot']:
            target = np.array([batch])
            target_deg = int(sum([degree[i].item() for i in target]))
            budget = int(min(round(target_deg / 2), round(degree.mean()))) if multi else 1
            ori = labels_np[target].item()
            best_wrong_label = 0
            one_order_nei = adj[target].nonzero()[1]
            tar_norm_adj = nor_adj_tensor[target.item()].to_dense()
            norm_a_target = tar_norm_adj[one_order_nei].unsqueeze(1)
            inj_feat, disc_score, masked_score_idx = model(target, one_order_nei, budget, feat, norm_a_target, node_emb,
                                                           W[ori], W[best_wrong_label],train_flag=training, eps=eps)

            new_num = num_decoder_model(inj_feat[16:24])
            new_number = (new_num * num_std + num_mean).abs().round()

            new_number[0] = torch.as_tensor(0).reshape(1, 1)
            new_number[1] = torch.as_tensor(100 if new_number[1] > 100 else new_number[1]).reshape(1, 1)
            new_number[2] = torch.as_tensor(
                15 if new_number[2] > 15 else (1 if new_number[2] < 1 else new_number[2])).reshape(1, 1)

            new_num = (new_number - num_mean) / num_std
            new_cat_float = cat_decoder_model(inj_feat[24:])
            new_cat = torch.tensor(0 if torch.all(new_cat_float) <= 0.5 else 1, device=device).reshape(1, 1)

            new_des_tensor = torch.cat((new_des_tensor, torch.zeros(1, 768).to(device)), 0)
            new_tweets_tensor = torch.cat((new_tweets_tensor, torch.zeros(1, 768).to(device)), 0)
            new_num_prop = torch.cat((new_num_prop, new_num.unsqueeze(0)), 0)
            new_category_prop = torch.cat((new_category_prop, new_cat), 0)
            new_edge_index = gen_extend_edge_index(new_edge_index, adj_tensor, disc_score, masked_score_idx, device)
            extend_edge_type = torch.tensor([1]).to(device)
            new_edge_type = torch.cat((new_edge_type, extend_edge_type), 0)
            new_logits = ori_model(new_des_tensor, new_tweets_tensor, new_num_prop, new_category_prop, new_edge_index,
                                   new_edge_type)
            new_logp = F.log_softmax(new_logits, dim=1)
            edge_nz = disc_score.detach().cpu().nonzero()

            if 0 == new_logp[target].argmax(1).item():
                names[dset + '_bot_atk'].append(0) #目标节点被检测为机器人
                new_labels = torch.cat((new_labels, torch.zeros(1).to(device)), 0)
                bot_count += 1
            else:
                names[dset + '_bot_atk'].append(1) #目标节点没有被检测为机器人
                new_labels = torch.cat((new_labels, torch.ones(1).to(device)), 0)
                human_count += 1
            del new_logits, new_logp
        print("被检测为机器人的注入节点数:", bot_count)
        print("被检测为人类的注入节点数:", human_count)

        # 把注入节点后的新图保存
        torch.save(new_des_tensor, "./attacked_processed_data/des_tensor.pt")
        torch.save(new_tweets_tensor, "./attacked_processed_data/tweets_tensor.pt")
        torch.save(new_num_prop, "./attacked_processed_data/num_properties_tensor.pt")
        torch.save(new_category_prop, "./attacked_processed_data/cat_properties_tensor.pt")
        torch.save(new_edge_index, "./attacked_processed_data/edge_index.pt")
        torch.save(new_edge_type, "./attacked_processed_data/edge_type.pt")
        torch.save(new_labels, "./attacked_processed_data/labels.pt")




if __name__ == '__main__':
    setup_seed(904)
    parser = argparse.ArgumentParser(description='GNIA')

    # configure
    parser.add_argument('--seed', type=int, default=904, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='GPU ID')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')

    # dataset
    parser.add_argument('--connect', default=False, type=bool, help='largest connected component')
    parser.add_argument('--multiedge', default=False, type=bool,
                        help='budget of malicious edges connected to injected node')

    # optimization
    parser.add_argument('--optimizer', choices=['Adam', 'RMSprop'], default='RMSprop', help='optimizer')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--wd', default=0., type=float, help='weight decay')
    parser.add_argument('--nepochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--patience', default=20, type=int, help='patience of early stopping')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize')

    # Hyperparameters
    parser.add_argument('--attrtau', default=None,
                        help='tau of gumbel softmax on attribute on discrete attributed graph')
    parser.add_argument('--edgetau', default=0.01, help='tau of gumbel softmax on edge')
    parser.add_argument('--epsdec', default=1, type=float, help='epsilon decay: coefficient of the gumbel sampling')
    parser.add_argument('--epsst', default=50, type=int, help='epsilon start: coefficient of the gumbel sampling')

    # Ignorable
    parser.add_argument('--counter', type=int, default=0, help='counter for recover training (Ignorable)')
    parser.add_argument('--best_score', type=float, default=0., help='best score for recover training (Ignorable)')
    parser.add_argument('--st_epoch', type=int, default=0, help='start epoch for recover training (Ignorable)')
    parser.add_argument('--local_rank', type=int, default=2, help='DDP local rank for parallel (Ignorable)')

    args = parser.parse_args()
    opts = args.__dict__.copy()
    # opts['discrete'] = False if 'k_' in opts['dataset'] else True
    opts['discrete'] = False
    print(opts)
    att_sucess = main(opts)
