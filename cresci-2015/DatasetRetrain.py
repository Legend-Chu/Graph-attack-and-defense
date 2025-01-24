import torch
import numpy as np
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from transformers import pipeline

class Twibot22(Dataset):
    def __init__(self, root='./attacked_processed_data/', device='cuda:0'):
        self.root = root
        self.device = device
        self.save = True

    def load_labels(self):
        print('Loading labels...', end='   ')
        labels = torch.load(self.root + "labels.pt").to(self.device)
        print('Finished')
        return labels

    def tweets_embedding(self):
        return torch.zeros((5800, 768)).to(self.device)

    def Des_embbeding(self):
        print('Running feature1 embedding')
        des_tensor = torch.load(self.root+"des_tensor.pt").to(self.device)
        print('Finished')
        return des_tensor

    def tweets_preprocess(self):
        print('Loading raw feature2...',end='   ')
        tweets = np.load('./processed_data/each_user_tweets.npy',allow_pickle=True)
        print('Finished')
        return tweets

    def num_prop_preprocess(self):
        print('Processing feature3...', end='   ')
        num_prop = torch.load(self.root + "num_properties_tensor.pt").to(self.device)
        print('Finished')
        return num_prop

    def cat_prop_preprocess(self):
        print('Processing feature4...', end='   ')
        category_properties = torch.load(self.root + "cat_properties_tensor.pt").to(self.device)
        print('Finished')
        return category_properties

    def Build_Graph(self):
        print('Building graph', end='   ')
        edge_index = torch.load(self.root + "edge_index.pt").to(self.device)
        edge_type = torch.load(self.root + "edge_type.pt").to(self.device)
        print('Finished')
        return edge_index, edge_type

    def train_val_test_mask(self):
        if self.root == './attacked_processed_data/':
            train_idx = range(3301)
            val_idx = range(3301, 4000)
            test_idx = range(4000, 5800)
        else:
            train_idx = torch.load(self.root + 'train_idx.pt').to(self.device)
            val_idx = torch.load(self.root + 'val_idx.pt').to(self.device)
            test_idx = torch.load(self.root + 'test_idx.pt').to(self.device)

        return train_idx, val_idx, test_idx

    def dataloader(self):
        labels = self.load_labels()
        des_tensor = self.Des_embbeding()
        self.tweets_preprocess()
        tweets_tensor = self.tweets_embedding()
        num_prop = self.num_prop_preprocess()
        category_prop = self.cat_prop_preprocess()
        edge_index, edge_type = self.Build_Graph()
        train_idx, val_idx, test_idx = self.train_val_test_mask()
        return des_tensor,tweets_tensor,num_prop,category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx