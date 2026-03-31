import torch
import numpy as np
import pandas as pd
import json
import os
from transformers import pipeline
from datetime import datetime as dt
from torch.utils.data import Dataset
from tqdm import tqdm


class Twibot22(Dataset):
    def __init__(self, root='./processed_data/', path='./sub_processed_data/', device='cpu'):
        self.root = root
        self.path = path
        self.device = device

    def load_labels(self):
        print('Loading labels...', end='   ')
        labels = torch.load(self.root + "label.pt").to(self.device)
        print('Finished')

        return labels

    def Des_embbeding(self):
        des_tensor = torch.load(self.root + "des_tensor.pt").to(self.device)
        return des_tensor

    def tweets_embedding(self):
        tweets_tensor = torch.load(self.root + "tweets_tensor.pt").to(self.device)
        return tweets_tensor

    def num_prop_preprocess(self):
        print('Processing feature3...', end='   ')
        num_prop = torch.load(self.root + "num_properties_tensor.pt").to(self.device)
        return num_prop

    def cat_prop_preprocess(self):
        print('Processing feature4...', end='   ')
        category_properties = torch.load(self.root + "cat_properties_tensor.pt").to(self.device)
        return category_properties

    def sub_feature(self):
        edge_index = torch.load(self.path + "edge_index.pt",map_location='cpu').to(self.device)
        edge_type = torch.load(self.path + "edge_type.pt",map_location='cpu').to(self.device)
        train_mask = torch.load(self.path + "train_idx.pt",map_location='cpu').to(self.device)
        val_mask = torch.load(self.path + "val_idx.pt",map_location='cpu').to(self.device)
        test_mask = torch.load(self.path + "test_idx.pt",map_location='cpu').to(self.device)
        subgraph_nodes = torch.load(self.path + "subgraph_nodes.pt",map_location='cpu').to(self.device)
        return edge_index, edge_type, train_mask, val_mask, test_mask, subgraph_nodes

    def dataloader(self):
        ori_labels = self.load_labels()
        ori_des_tensor = self.Des_embbeding()
        ori_tweets_tensor = self.tweets_embedding()
        ori_num_prop = self.num_prop_preprocess()
        ori_category_prop = self.cat_prop_preprocess()
        edge_index, edge_type, train_mask, val_mask, test_mask, subgraph_nodes = self.sub_feature()

        des_tensor = ori_des_tensor[subgraph_nodes]
        tweets_tensor = ori_tweets_tensor[subgraph_nodes]
        num_prop = ori_num_prop[subgraph_nodes]
        category_prop = ori_category_prop[subgraph_nodes]
        labels = ori_labels[subgraph_nodes]

        new_ids = torch.arange(len(subgraph_nodes)).to(self.device)

        train_mask = torch.tensor([new_ids[subgraph_nodes == i] for i in train_mask])
        val_mask = torch.tensor([new_ids[subgraph_nodes == i] for i in val_mask])
        test_mask = torch.tensor([new_ids[subgraph_nodes == i] for i in test_mask])

        edge_index[0,:] = torch.tensor([new_ids[subgraph_nodes == i] for i in edge_index[0,:]])
        edge_index[1,:] = torch.tensor([new_ids[subgraph_nodes == i] for i in edge_index[1,:]])

        return des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_mask, val_mask, test_mask
