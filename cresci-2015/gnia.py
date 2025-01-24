import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(logits, tau, random_flag=False, eps=0, dim=-1):
    if random_flag:
        uniform_rand = torch.rand_like(logits)
        epsilon = torch.zeros_like(uniform_rand) + 1e-6
        nz_uniform_rand = torch.where(uniform_rand<=0, epsilon, uniform_rand)
        gumbels = -(-(nz_uniform_rand.log())).log()

        tau = 0.01
        gumbels = (logits + eps * gumbels) / tau
        output = gumbels.softmax(dim)
    else:
        tau = 0.01
        output = logits/(0.01*tau)
        output = output.softmax(dim)
    return output

def gumbel_topk(logits, budget, tau, random_flag, eps, device):
    mask = torch.zeros(logits.shape).to(device)
    idx = np.arange(logits.shape[0])
    discrete = torch.zeros_like(logits).to(device)
    discrete.requires_grad_()
    for i in range(budget):
        if i != 0:
            tmp_score, tmp_idx = torch.max(tmp, dim=0)
            mask[tmp_idx] = 9999 
        cur_discrete = logits - mask
        
        tmp = gumbel_softmax(cur_discrete, tau, random_flag, eps)
        discrete = discrete + tmp
    return discrete

# --------------------------- MLP ----------------------------
# MLP    
class MLP(nn.Module):
    def __init__(self, input_dim, hid1,hid2, output_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_dim, hid1)
        self.l2 = nn.Linear(hid1, hid2)
        self.l3 = nn.Linear(hid2, output_dim)
        # self.dropout = dropout

        nn.init.kaiming_normal_(self.l1.weight)
        nn.init.kaiming_normal_(self.l2.weight)
        nn.init.kaiming_normal_(self.l3.weight)

        self.fc1 = nn.Sequential(
            self.l1,
            nn.LeakyReLU(),
            self.l2,
            nn.LeakyReLU(),
            self.l3
        )
        
    def forward(self, x):
        output = self.fc1(x)
        return output


# ------------------------- Generalizable Node Injection Attack (G-NIA) model ----------------------------
# Attribute generation
class AttrGeneration(nn.Module):
    def __init__(self, labels, tau, feat_dim, weight, discrete, device, tar_num, feat_max, feat_min):
        super(AttrGeneration, self).__init__()
        self.labels = labels
        self.label_dim = labels.max().item() + 1
        self.feat_dim = feat_dim
        self.weight = weight.t()
        self.discrete = discrete
        self.tau = tau
        self.device = device
        self.tar_num = tar_num
        self.feat_max = feat_max
        self.feat_min = feat_min
        self.obtain_feat = MLP(3*self.label_dim+2*self.feat_dim, 128, 512, self.feat_dim)
    
    def pool_func(self, wlabel, wsec):
        sub_graph_emb = self.node_emb[self.sub_graph_nodes].mean(0)

        tarfeat_emb = F.relu(torch.mm(self.feat[self.target], self.weight))

        if self.tar_num == 1:
            graph_emb = torch.cat((sub_graph_emb.unsqueeze(0), self.node_emb[self.target], tarfeat_emb, wlabel, wsec), 1)
        else:
            tar_emb = self.node_emb[self.target].mean(0).unsqueeze(0)
            tarfeat_emb = tarfeat_emb.mean(0).unsqueeze(0)
            graph_emb = torch.cat((sub_graph_emb.unsqueeze(0), tar_emb, tarfeat_emb, wlabel.mean(0).unsqueeze(0), wsec.mean(0).unsqueeze(0)), 1)
        return graph_emb

    def forward(self, target, feat, sub_graph_nodes, node_emb, wlabel, wsec, feat_num=None,  eps=1, train_flag=False):
        self.target = target
        self.node_emb = node_emb
        self.sub_graph_nodes = sub_graph_nodes
        self.feat = feat
        self.graph_embed = self.pool_func(wlabel, wsec)
        self.add_feat = self.obtain_feat(self.graph_embed).squeeze(0)
        if self.discrete == True:
            inj_feat = gumbel_topk(self.add_feat, feat_num, self.tau, train_flag, eps, self.device)
        else:
            inj_feat = self.add_feat.sigmoid()
            inj_feat = (self.feat_max - self.feat_min) * inj_feat + self.feat_min
        new_feat = torch.cat((self.feat, inj_feat.unsqueeze(0)), 0)

        return new_feat, inj_feat


# Edge generation
class EdgeGeneration(nn.Module):
    def __init__(self, labels, feat_dim, weight, device, tar_num=1, tau=None):
        super(EdgeGeneration, self).__init__()  # 调用父类的初始化函数
        self.labels = labels  # 保存标签数据
        self.label_dim = self.labels.max() + 1  # 获取标签的最大值并加1，用于确定标签维度
        self.feat_dim = feat_dim  # 保存特征的维度
        self.weight = weight.t()  # 保存转置后的权重矩阵
        self.tar_num = tar_num  # 保存目标节点的数量
        # 初始化一个多层感知机（MLP）用于计算边的分数，输入维度为（3倍标签维度 + 2倍特征维度 + 目标节点数量），隐藏层大小为512和32，输出为1
        self.obtain_score = MLP(3 * self.label_dim + 2 * self.feat_dim + tar_num, 512, 32, 1)
        self.tau = tau  # 保存温度参数，用于控制Gumbel-Softmax的随机性
        self.device = device  # 保存当前使用的计算设备

    def concat(self, new_feat, wlabel, wsec):
        # 计算子图中所有节点的新特征与权重矩阵的乘积
        sub_xw = torch.mm(new_feat[self.sub_graph_nodes], self.weight)
        # 计算目标节点的新特征与权重矩阵的乘积
        tar_xw = torch.mm(new_feat[self.target], self.weight)
        # 计算新添加节点的新特征与权重矩阵的乘积
        add_xw = torch.mm(new_feat[-1].unsqueeze(0), self.weight)

        # 将新添加节点的特征复制到与子图节点数量相同的维度
        add_xw_rep = add_xw.repeat(len(self.sub_graph_nodes), 1)

        # 如果目标节点数量为1
        if self.tar_num == 1:
            # 如果邻接矩阵是稀疏矩阵
            if self.adj_tensor.is_sparse:
                tar_norm_adj = self.adj_tensor[self.target.item()].to_dense()  # 获取目标节点的邻接矩阵行并转为密集矩阵
                norm_a_target = tar_norm_adj[self.sub_graph_nodes].unsqueeze(1)  # 获取子图中节点的邻接关系并增加一个维度
            # 如果邻接矩阵的形状只有一列
            elif self.adj_tensor.shape[1] == 1:
                norm_a_target = self.adj_tensor  # 使用原始邻接矩阵
            else:
                norm_a_target = self.adj_tensor[self.sub_graph_nodes, self.target].unsqueeze(
                    0).t()  # 获取子图中节点与目标节点之间的邻接关系并转置
        else:  # 如果目标节点数量大于1
            if self.adj_tensor.is_sparse:  # 如果邻接矩阵是稀疏矩阵
                self.adj_tensor = self.adj_tensor.to_dense()  # 将邻接矩阵转为密集矩阵
            # 获取每个目标节点与子图中节点的邻接关系，并将其连接在一起
            norm_a_targets_list = [self.adj_tensor[self.sub_graph_nodes, target].unsqueeze(0).t() for target in
                                   self.target]
            norm_a_target = torch.cat(norm_a_targets_list, 1)

        # 如果目标节点数量为1
        if self.tar_num == 1:
            tar_xw_rep = tar_xw.repeat(len(self.sub_graph_nodes), 1)  # 复制目标节点特征到与子图节点数量相同的维度
            w_rep = wlabel.repeat(len(self.sub_graph_nodes), 1)  # 复制标签到与子图节点数量相同的维度
            w_sec_rep = wsec.repeat(len(self.sub_graph_nodes), 1)  # 复制次级标签到与子图节点数量相同的维度
        else:  # 如果目标节点数量大于1
            tar_xw_rep = tar_xw.mean(0).repeat(len(self.sub_graph_nodes), 1)  # 计算目标节点特征的平均值并复制
            w_rep = wlabel.mean(0).repeat(len(self.sub_graph_nodes), 1)  # 计算标签的平均值并复制
            w_sec_rep = wsec.mean(0).repeat(len(self.sub_graph_nodes), 1)  # 计算次级标签的平均值并复制

        # 将计算得到的各种特征、邻接关系、标签和次级标签连接起来，作为模型输出
        concat_output = torch.cat((tar_xw_rep, sub_xw, add_xw_rep, norm_a_target, w_rep, w_sec_rep), 1)
        return concat_output  # 返回连接后的输出

    def forward(self, budget, target, sub_graph_nodes, new_feat, adj_tensor, wlabel, wsec, eps=0, train_flag=False):
        self.budget = budget  # 保存预算（即可以增加的边数）
        self.adj_tensor = adj_tensor  # 保存邻接矩阵
        self.sub_graph_nodes = sub_graph_nodes  # 保存子图中的节点
        self.target = target  # 保存目标节点
        self.sub_cat_addnode = self.concat(new_feat, wlabel, wsec)  # 调用concat函数生成模型输入
        self.output = self.obtain_score(self.sub_cat_addnode).transpose(0, 1)  # 通过MLP计算输出分数并转置

        if self.output.dim() > 1:  # 如果输出的维度大于1
            self.output = self.output.squeeze(0)  # 压缩输出张量的第0维
        elif self.output.dim() == 0:  # 如果输出的维度为0
            self.output = self.output.unsqueeze(0)  # 扩展输出张量的维度

        # 使用Gumbel-Softmax对得分进行采样，得到离散的边得分
        score = gumbel_topk(self.output, budget, self.tau, train_flag, eps, self.device)
        # 将子图中的节点索引转为长整型张量，并增加一个维度
        score_idx = torch.LongTensor(sub_graph_nodes.reshape(sub_graph_nodes.shape[0])).unsqueeze(0)
        return score, score_idx  # 返回采样得到的边得分和对应的节点索引


class GNIA(nn.Module):  # 定义一个继承自PyTorch的神经网络模块类GNIA
    def __init__(self, labels, feat_dim, weight, discrete, device, tar_num=1, feat_max=None, feat_min=None,
                 feat_num=None, attr_tau=None, edge_tau=None):
        super(GNIA, self).__init__()  # 调用父类的初始化函数
        self.labels = labels      # 保存标签数据
        self.feat_dim = feat_dim  # 保存特征的维度
        self.feat_num = feat_num  # 保存需要生成的特征数量
        # 初始化并保存节点生成器，传入相关的参数：标签、属性温度、特征维度、权重、是否离散化、设备、目标数量、特征最大最小值
        self.add_node_agent = AttrGeneration(self.labels, attr_tau, self.feat_dim, weight, discrete, device, tar_num,
                                             feat_max, feat_min).to(device)
        # 初始化并保存边生成器，传入相关的参数：标签、特征维度、权重、设备、目标数量、边温度
        self.add_edge_agent = EdgeGeneration(self.labels, feat_dim, weight, device, tar_num, edge_tau)
        self.tar_num = tar_num  # 保存目标节点数量
        self.discrete = discrete  # 保存是否进行离散化的标志
        self.device = device  # 保存当前使用的计算设备

    def add_node_and_update(self, feat_num, wlabel, wsec, eps=0, train_flag=False):
        # 调用节点生成器，生成新的节点特征并更新图
        return self.add_node_agent(self.target, self.feat, self.sub_graph_nodes, self.node_emb, wlabel, wsec, feat_num,
                                   eps, train_flag)

    def add_edge_and_update(self, new_feat, wlabel, wsec, eps=0, train_flag=False):
        # 调用边生成器，生成新的边并更新图
        return self.add_edge_agent(self.budget, self.target, self.sub_graph_nodes, new_feat, self.nor_adj_tensor,
                                   wlabel, wsec, eps, train_flag)

    def forward(self, target, sub_graph_nodes, budget, feat, nor_adj_tensor, node_emb, wlabel, wsec, train_flag, eps=0):
        self.target = target  # 保存目标节点
        self.nor_adj_tensor = nor_adj_tensor  # 保存归一化后的邻接矩阵张量
        self.sub_graph_nodes = sub_graph_nodes  # 保存子图中的节点
        self.budget = budget  # 保存预算，即可以增加的边数或节点数
        self.feat = feat  # 保存图中的特征矩阵

        self.n = self.feat.shape[0]  # 获取特征矩阵的节点数
        self.node_emb = node_emb  # 保存节点的嵌入表示

        if self.tar_num == 1:  # 如果目标节点数量为1
            wlabel = wlabel.unsqueeze(0)  # 对标签进行维度扩展
            wsec = wsec.unsqueeze(0)  # 对次级标签进行维度扩展

        # 调用函数生成新的节点特征，并更新图结构
        self.new_feat, self.add_feat = self.add_node_and_update(self.feat_num, wlabel, wsec, eps, train_flag=train_flag)

        # 调用函数生成新的边，并更新图结构
        self.score, self.masked_score_idx = self.add_edge_and_update(self.new_feat, wlabel, wsec, eps=eps,
                                                                     train_flag=train_flag)

        # 评估阶段
        if train_flag:  # 如果在训练过程中
            self.disc_score = self.score  # 使用生成的边的得分
        else:  # 如果在评估过程中
            if self.discrete:  # 如果需要离散化
                feat_values, feat_indices = self.add_feat.topk(self.feat_num)  # 选择得分最高的特征
                self.disc_feat = torch.zeros_like(self.add_feat).to(self.device)  # 创建一个与add_feat形状相同的全零张量
                self.disc_feat[feat_indices] = 1.  # 对最高得分的特征进行标记
                self.new_feat[-1] = self.disc_feat  # 更新最后一个节点的特征

            edge_values, edge_indices = self.score.topk(budget)  # 选择得分最高的边
            self.disc_score = torch.zeros_like(self.score).to(self.device)  # 创建一个与score形状相同的全零张量
            self.disc_score[edge_indices] = 1.  # 对最高得分的边进行标记

        # 返回生成的特征、边的得分和掩码索引
        return self.add_feat, self.disc_score, self.masked_score_idx


