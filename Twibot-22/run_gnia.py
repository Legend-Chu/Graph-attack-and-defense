import sys
import os
import math
import argparse
import numpy as np
import torch.utils.data as Data
np_load_old = np.load
np.aload = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
from gnia import GNIA
sys.path.append('..')
from utils import *
from model import oriRGCN, encoder, RGCN, num_decoder, cat_decoder, RGCN_weight
from subDataset import Twibot22
import torch
from torch import nn
from utils import accuracy
import numpy as np

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

    ckpt_save_dirs = 'checkpoint/bot_gnia/'
    model_save_file = ckpt_save_dirs
    if not os.path.exists(ckpt_save_dirs):
        os.makedirs(ckpt_save_dirs)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing data
    root = './processed_data/'
    path = './sub_processed_data/'
    dataset = Twibot22(root=root, path=path, device=device)
    des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels_np, train_mask, val_mask, test_mask = dataset.dataloader()

    train_mask_human = []
    train_mask_bot = []
    for i in train_mask:
        if labels_np[i] == 0:
            train_mask_human.append(i)
        else:
            train_mask_bot.append(i)

    val_mask_human = []
    val_mask_bot = []
    for i in val_mask:
        if labels_np[i] == 0:
            val_mask_human.append(i)
        else:
            val_mask_bot.append(i)

    test_mask_human = []
    test_mask_bot = []
    for i in test_mask:
        if labels_np[i] == 0:
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

    ori_model = oriRGCN().to(device)
    ori_model.load_state_dict(torch.load('model/model_ori_rgcn.pth'))

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

    num_decoder_model = num_decoder().to(device)
    num_decoder_model.load_state_dict(torch.load('model/model_num_decoder.pth', map_location='cpu'))
    cat_decoder_model = cat_decoder().to(device)
    cat_decoder_model.load_state_dict(torch.load('model/model_cat_decoder.pth', map_location='cpu'))

    num_dict = np.load('processed_data/num_properties_pytmeanstd.npy', allow_pickle=True).item()
    num_mean = torch.tensor(
        [num_dict['followers_count_mean'], num_dict['active_days_mean'], num_dict['screen_name_length_mean'],
         num_dict['following_count_mean'], num_dict['statues_mean']], device=device)
    num_std = torch.tensor(
        [num_dict['followers_count_std'], num_dict['active_days_std'], num_dict['screen_name_length_std'],
         num_dict['following_count_std'], num_dict['statues_std']], device=device)

    features = encoder_model(des_tensor, tweets_tensor, num_prop, category_prop)
    features = torch.tensor(features, device='cpu')
    features = np.array(features)
    features = sp.csr_matrix(features, dtype=np.float32)

    adj = load_adj(edge_index)
    n = adj.shape[0]
    nc = labels_np.max() + 1
    nfeat = features.shape[1]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(n)
    adj[adj > 1] = 1

    print('adj shape:', adj.shape)

    if connect:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels_np = labels_np[lcc]
        n = adj.shape[0]
        print('Nodes num:', n)

    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    nor_adj_tensor = normalize_tensor(adj_tensor)

    feat = torch.from_numpy(features.todense().astype('double')).float().to(device)
    feat_max = feat.max(0).values
    feat_min = feat.min(0).values
    labels = labels_np
    degree = adj.sum(1)
    deg = torch.FloatTensor(degree).flatten().to(device)
    feat_num = int(features.sum(1).mean())
    eps_threshold = [epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps / epsilon_decay) for steps in range(total_steps)]

    detect_model = RGCN().to(device)
    rgcn_weight = model_dict['rgcn.weight'].clone().detach()
    rgcn_root = model_dict['rgcn.root'].clone().detach()
    RGCN_dict = {'rgcn.weight': rgcn_weight, 'rgcn.root': rgcn_root}
    detect_model.load_state_dict(RGCN_dict)

    detect_model.eval()
    for p in detect_model.parameters():
        p.requires_grad = False

    node_emb = detect_model(feat, edge_index, edge_type)

    RGCN_weight_model = RGCN_weight().to(device)

    W_weight = detect_model.rgcn.weight.data.detach()
    W1 = W_weight[0]
    W2 = W_weight[1]
    W = torch.cat((W1, W2), dim=0).t()
    W = RGCN_weight_model(W)

    torch.save(RGCN_weight_model.state_dict(), 'model/model_RGCN_weight.pth')

    logits = detect_model(feat, edge_index, edge_type)
    logp = F.log_softmax(logits, dim=1)
    acc = accuracy(logp, labels)
    print('Acc:',acc)
    print('Train Acc:',accuracy(logp[train_mask], labels[train_mask]))
    print('Valid Acc:',accuracy(logp[val_mask], labels[val_mask]))
    print('Test Acc:',accuracy(logp[test_mask], labels[test_mask]))
    print('*' * 30)

    # Initialization
    model = GNIA(labels, nfeat, W, discrete, device, feat_min=feat_min, feat_max=feat_max, feat_num=feat_num, attr_tau=attr_tau, edge_tau=edge_tau).to(device)
    stopper = EarlyStop_loss(patience=patience)

    if opts['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
    elif opts['optimizer'] == 'RMSprop':
        optimizer = torch.optim.RMSprop([{'params': model.parameters()}], lr=lr, weight_decay=0)
    else:
        raise ValueError('Unsupported argument for the optimizer')

    x = torch.LongTensor(train_mask_bot)
    y = labels[train_mask_bot].to(torch.device('cpu'))
    torch_dataset = Data.TensorDataset(x, y)
    # train_sampler = Data.distributed.DistributedSampler(
    #     torch_dataset,
    #     num_replicas=2,
    #     rank=local_rank,
    # )
    # batch_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, num_workers=24, sampler=train_sampler)
    batch_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, num_workers=0)

    if st_epoch != 0:
        model.load_state_dict(torch.load(model_save_file+'checkpoint1.pt'))
        stopper.best_score = best_score
        stopper.counter = counter
    # Training and Validation
    for epoch in range(st_epoch, nepochs):
        training = True
        print("Epoch {:05d}".format(epoch))
        train_atk_success = []
        val_atk_success = []
        train_loss_arr = []
        eps = eps_threshold[epoch] if epoch < len(eps_threshold) else eps_threshold[-1]

        for batch_x,_ in batch_loader:
            loss_arr = []

            for train_batch in batch_x:
                try:
                    target = np.array([train_batch])
                    target_deg = int(sum([degree[i].item() for i in target]))
                    budget = int(min(round(target_deg/2), round(degree.mean()))) if multi else 1
                    ori = labels_np[target].item()
                    best_wrong_label = 0
                    one_order_nei = adj[target].nonzero()[1]

                    tar_norm_adj = nor_adj_tensor[target.item()].to_dense()
                    norm_a_target = tar_norm_adj[one_order_nei].unsqueeze(1)

                    inj_feat, disc_score, masked_score_idx = model(target, one_order_nei, budget, feat, norm_a_target,
                                                               node_emb,
                                                               W[ori], W[best_wrong_label], train_flag=training,
                                                               eps=eps)
                    new_num = num_decoder_model(inj_feat[16:24])
                    new_cat = cat_decoder_model(inj_feat[24:])
                    new_cat[0] = torch.as_tensor(0 if new_cat[0] <= 0.5 else 1, device=device).reshape(1, 1)
                    new_cat[1] = torch.as_tensor(0 if new_cat[1] <= 0.5 else 1, device=device).reshape(1, 1)
                    new_cat[2] = torch.as_tensor(0 if new_cat[2] <= 0.5 else 1, device=device).reshape(1, 1)
                    new_des_tensor = torch.cat((des_tensor, torch.zeros(1, 768).to(device)), 0)
                    new_tweets_tensor = torch.cat((tweets_tensor, torch.zeros(1, 768).to(device)), 0)
                    new_num_prop = torch.cat((num_prop, new_num.unsqueeze(0)), 0)
                    new_category_prop = torch.cat((category_prop, new_cat.unsqueeze(0)), 0)
                    new_edge_index = gen_extend_edge_index(edge_index, adj_tensor, disc_score, masked_score_idx, device)

                    extend_edge_type = torch.tensor([1]).to(device)
                    new_edge_type = torch.cat((edge_type, extend_edge_type), 0)

                    new_logits = ori_model(new_des_tensor,new_tweets_tensor,new_num_prop,new_category_prop, new_edge_index, new_edge_type)  # (10005,41)
                except IndexError:
                    continue

                else:
                    loss = F.relu(new_logits[target, ori] - new_logits[target, best_wrong_label])
                    new_logp = F.log_softmax(new_logits, dim=1)
                    train_atk_success.append(int(0 == new_logits[target].argmax(1).item()))
                    loss_arr.append(loss)
            if len(loss_arr) > 0:
                train_loss = sum(loss_arr)
                optimizer.zero_grad()
                train_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.1)
                optimizer.step()
                train_loss_arr.append((train_loss / len(loss_arr)).detach().cpu().item())
            else:
                print("Empty loss array for this batch, skipping backpropagation.")
            del loss_arr
            if 'train_loss' in locals():
                del train_loss
        if len(train_loss_arr) > 0:
            print('Training set Loss:', np.array(train_loss_arr).mean())
        else:
            print('Training set Loss: No valid losses recorded for this epoch.')
        print('Training set: Attack success rate:', np.array(train_atk_success).mean())
        del train_atk_success
        if 'train_loss_arr' in locals():
            del train_loss_arr
        torch.cuda.empty_cache()


        val_loss_arr = []
        training = False
        for val_batch in val_mask_bot:
            target = np.array([val_batch])
            target_deg = int(sum([degree[i].item() for i in target]))
            budget = int(min(round(target_deg/2), round(degree.mean()))) if multi else 1
            ori = labels_np[target].item()
            best_wrong_label = 0
            one_order_nei = adj[target].nonzero()[1]
            tar_norm_adj = nor_adj_tensor[target.item()].to_dense()
            norm_a_target = tar_norm_adj[one_order_nei].unsqueeze(1)
            inj_feat, disc_score, masked_score_idx = model(target, one_order_nei, budget, feat, norm_a_target, node_emb,
                                                           W[ori], W[best_wrong_label], train_flag=training,
                                                           eps=eps)
            new_num = num_decoder_model(inj_feat[16:24])
            new_cat = cat_decoder_model(inj_feat[24:])
            new_cat[0] = torch.as_tensor(0 if new_cat[0] <= 0.5 else 1, device=device).reshape(1, 1)
            new_cat[1] = torch.as_tensor(0 if new_cat[1] <= 0.5 else 1, device=device).reshape(1, 1)
            new_cat[2] = torch.as_tensor(0 if new_cat[2] <= 0.5 else 1, device=device).reshape(1, 1)
            new_des_tensor = torch.cat((des_tensor, torch.zeros(1, 768).to(device)), 0)
            new_tweets_tensor = torch.cat((tweets_tensor, torch.zeros(1, 768).to(device)), 0)
            new_num_prop = torch.cat((num_prop, new_num.unsqueeze(0)), 0)
            new_category_prop = torch.cat((category_prop, new_cat.unsqueeze(0)), 0)
            new_edge_index = gen_extend_edge_index(edge_index, adj_tensor, disc_score, masked_score_idx, device)

            extend_edge_type = torch.tensor([1]).to(device)
            new_edge_type = torch.cat((edge_type, extend_edge_type), 0)

            new_logits = ori_model(new_des_tensor, new_tweets_tensor, new_num_prop, new_category_prop,
                                   new_edge_index, new_edge_type)
            loss = F.relu(new_logits[target, ori] - new_logits[target, best_wrong_label])
            new_logp = F.log_softmax(new_logits, dim=1)
            val_atk_success.append(int(0 == new_logits[target].argmax(1).item()))
            val_loss_arr.append(loss.detach().cpu().item())
        print('Validation set Loss:', np.array(val_loss_arr).mean())
        print('Validation set: Attack success rate:', np.array(val_atk_success).mean())

        val_loss = np.array(val_loss_arr).mean()

        if stopper.step(val_loss, model, model_save_file):
            break
        del val_loss_arr, val_atk_success
        torch.cuda.empty_cache()


    # Test Part
    names = locals()
    training = False
    model.load_state_dict(torch.load(model_save_file + 'checkpoint.pt'))
    for p in model.parameters():
        p.requires_grad = False
    for dset in ['test']:
        names[dset + '_bot_atk'] = []
        names[dset + '_new_node'] = []
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
                                                           W[ori], W[best_wrong_label], train_flag=training, eps=eps)
            new_num = num_decoder_model(inj_feat[16:24])
            new_cat = cat_decoder_model(inj_feat[24:])
            new_cat[0] = torch.as_tensor(0 if new_cat[0] <= 0.5 else 1, device=device).reshape(1, 1)
            new_cat[1] = torch.as_tensor(0 if new_cat[1] <= 0.5 else 1, device=device).reshape(1, 1)
            new_cat[2] = torch.as_tensor(0 if new_cat[2] <= 0.5 else 1, device=device).reshape(1, 1)
            new_des_tensor = torch.cat((des_tensor, torch.zeros(1, 768).to(device)), 0)
            new_tweets_tensor = torch.cat((tweets_tensor, torch.zeros(1, 768).to(device)), 0)
            new_num_prop = torch.cat((num_prop, new_num.unsqueeze(0)), 0)
            new_category_prop = torch.cat((category_prop, new_cat.unsqueeze(0)), 0)
            new_edge_index = gen_extend_edge_index(edge_index, adj_tensor, disc_score, masked_score_idx, device)

            extend_edge_type = torch.tensor([1]).to(device)
            new_edge_type = torch.cat((edge_type, extend_edge_type), 0)

            new_logits = ori_model(new_des_tensor, new_tweets_tensor, new_num_prop, new_category_prop,
                                   new_edge_index, new_edge_type)
            out_tar = target

            new_logp = F.log_softmax(new_logits, dim=1)
            edge_nz = disc_score.detach().cpu().nonzero()
            # if discrete:
            #     print('\t Attribute:', feat_nz.shape[0], feat_nz.squeeze())
            # print('\t Edge:', edge_nz.shape[0], edge_nz.squeeze())
            # print('\t pred: %d, sec: %d, label: %d' % (new_logits[out_tar].argmax(), best_wrong_label, ori))

            if 0 == new_logp[out_tar].argmax(1).item():
                names[dset + '_bot_atk'].append(1)
            else:
                names[dset + '_bot_atk'].append(0)

            new_node_tar = np.array([-1])

            if 1 == new_logits[new_node_tar].argmax(1).item():
                names[dset + '_new_node'].append(1)
            else:
                names[dset + '_new_node'].append(0)

            del new_logits, new_logp

        print('Hidden Bot Ratio of ' + dset + ' set:', np.array(names[dset + '_bot_atk']).mean())
        print('New Node Become Bot Ratio of ' + dset + ' set:', np.array(names[dset + '_new_node']).mean())
        print('*' * 30)

    # np.save('useful_output/' + surro_type + '_gene/' + dataset + '_' + suffix + '_featsum.npy', np.array(feat_sum))
    # np.save(output_save_dirs + '_' + suffix + '_atk_success.npy', np.array(atk_suc))


if __name__ == '__main__':
    setup_seed(1)
    parser = argparse.ArgumentParser(description='GNIA')

    # configure
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='GPU ID')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')

    # dataset
    parser.add_argument('--connect', default=False, type=bool, help='largest connected component')
    parser.add_argument('--multiedge', default=False, type=bool, help='budget of malicious edges connected to injected node')

    # optimization
    parser.add_argument('--optimizer', choices=['Adam', 'RMSprop'], default='RMSprop', help='optimizer')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--wd', default=0., type=float, help='weight decay')
    parser.add_argument('--nepochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--patience', default=10, type=int, help='patience of early stopping')
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