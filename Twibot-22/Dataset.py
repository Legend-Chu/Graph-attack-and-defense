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
    def __init__(self, root='./processed_data/', device='cpu', process=True, save=True):
        self.root = root
        self.device = device
        self.save = save
        self.process = process

    def load_labels(self):
        print('Loading labels...', end='   ')
        path = self.root + 'label.pt'
        labels = torch.load(path).to(self.device)
        print('Finished')
        return labels

    def Des_Preprocess(self):
        print('Loading raw feature1...', end='   ')
        path = self.root + 'description.npy'
        if not os.path.exists(path):
            description = []
            for i in range(self.df_data.shape[0]):
                if self.df_data['profile'][i] is None or self.df_data['profile'][i]['description'] is None:
                    description.append('None')
                else:
                    description.append(self.df_data['profile'][i]['description'])
            description = np.array(description)
            if self.save:
                np.save(path, description)
        else:
            description = np.load(path, allow_pickle=True)
        print('Finished')
        return description

    def Des_embbeding(self):
        print('Running feature1 embedding')
        path = self.root + "des_tensor.pt"
        des_tensor = torch.load(path).to(self.device)
        print('Finished')
        return des_tensor

    def tweets_preprocess(self):
        print('Loading raw feature2...', end='   ')
        path = self.root + 'tweets.npy'
        if not os.path.exists(path):
            tweets = []
            for i in range(self.df_data.shape[0]):
                one_usr_tweets = []
                if self.df_data['tweet'][i] is None:
                    one_usr_tweets.append('')
                else:
                    for each in self.df_data['tweet'][i]:
                        one_usr_tweets.append(each)
                tweets.append(one_usr_tweets)
            tweets = np.array(tweets)
            if self.save:
                np.save(path, tweets)
        else:
            tweets = np.load(path, allow_pickle=True)
        print('Finished')
        return tweets

    def tweets_embedding(self):
        print('Running feature2 embedding')
        path = self.root + "tweets_tensor.pt"
        tweets_tensor = torch.load(path).to(self.device)
        print('Finished')
        return tweets_tensor

    def num_prop_preprocess(self):
        print('Processing feature3...', end='   ')
        path = self.root + 'num_properties_tensor.pt'
        num_prop = torch.load(path).to(self.device)
        print('Finished')
        return num_prop

    def cat_prop_preprocess(self):
        print('Processing feature4...', end='   ')
        path = self.root + 'cat_properties_tensor.pt'
        category_properties = torch.load(path).to(self.device)
        print('Finished')
        return category_properties

    def Build_Graph(self):
        print('Building graph', end='   ')
        edge_index = torch.load(self.root + "edge_index.pt").to(self.device)
        edge_type = torch.load(self.root + "edge_type.pt").to(self.device)
        print('Finished')
        return edge_index, edge_type

    def train_val_test_mask(self):
        if self.root == './processed_data/':
            train_idx = torch.load(self.root + 'train_idx.pt')
            val_idx = torch.load(self.root + 'val_idx.pt')
            test_idx = torch.load(self.root + 'test_idx.pt')
        else:
            train_idx = range(8278)
            val_idx = range(8278, 8278 + 2365)
            test_idx = range(8278 + 2365, 8278 + 2365 + 1183)

        return train_idx, val_idx, test_idx

    def dataloader(self):
        labels = self.load_labels()
        # self.Des_Preprocess()
        des_tensor = self.Des_embbeding()
        # self.tweets_preprocess()
        tweets_tensor = self.tweets_embedding()
        num_prop = self.num_prop_preprocess()
        category_prop = self.cat_prop_preprocess()
        edge_index, edge_type = self.Build_Graph()
        train_idx, val_idx, test_idx = self.train_val_test_mask()
        return des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx