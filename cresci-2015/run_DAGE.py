from model import BotRGCN, oriRGCN, HGTDetector, SHGNDetector,RGCNDetector,BotGCN
from DatasetRetrain import Twibot22
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

root='./attacked_processed_data/'

dataset = Twibot22(root=root, device=device)
(des_tensor, tweet_channel, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx) = dataset.dataloader()
