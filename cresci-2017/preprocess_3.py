import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, HeteroData
from tqdm import tqdm
from datetime import datetime as dt
import sys
from dataset_tool import fast_merge,df_to_mask
print('loading raw data')
node = pd.read_json("../datasets/cresci-2015/node.json")
print('processing raw data')
user, tweet = fast_merge(dataset='cresci-2015')
path = 'processed_data/'

# num_properties
print('extracting num_properties')

# 关注数
following_count = []
for i, each in enumerate(node['public_metrics']):
    if i == len(user):
        break
    if each is not None and isinstance(each, dict):
        if each['following_count'] is not None:
            following_count.append(each['following_count'])
        else:
            following_count.append(0)
    else:
        following_count.append(0)

#
statues = []
for i, each in enumerate(node['public_metrics']):
    if i == len(user):
        break
    if each is not None and isinstance(each, dict):
        if each['listed_count'] is not None:
            statues.append(each['listed_count'])
        else:
            statues.append(0)
    else:
        statues.append(0)

# 粉丝数
followers_count = []
for each in user['public_metrics']:
    if each is not None and each['followers_count'] is not None:
        followers_count.append(int(each['followers_count']))
    else:
        followers_count.append(0)

screen_name_length = []
for each in user['username']:
    if each is not None:
        screen_name_length.append(len(each))
    else:
        screen_name_length.append(int(0))

# 创建时间
created_at=user['created_at']
print(created_at)
created_at=pd.to_datetime(created_at,unit='s')

date0 = dt.strptime('Tue Sep 1 00:00:00 +0000 2020 ', '%a %b %d %X %z %Y ')
active_days = []
for each in created_at:
    active_days.append((date0 - each).days)

df_followers_count = pd.DataFrame(followers_count)
followers_count_mean = df_followers_count.mean()
followers_count_std = df_followers_count.std()

df_active_days = pd.DataFrame(active_days)
active_days_mean = df_active_days.mean()
active_days_std = df_active_days.std()

df_screen_name_length = pd.DataFrame(screen_name_length)
screen_name_length_mean = df_screen_name_length.mean()
screen_name_length_std = df_screen_name_length.std()

df_following_count = pd.DataFrame(following_count)
following_count_mean = df_following_count.mean()
following_count_std = df_following_count.std()

df_statues = pd.DataFrame(statues)
statues_mean = df_statues.mean()
statues_std = df_statues.std()


followers_count = torch.tensor(followers_count)
active_days = torch.tensor(active_days)
screen_name_length = torch.tensor(screen_name_length)
following_count = torch.tensor(following_count)
statues = torch.tensor(statues)

followers_count = torch.unsqueeze(followers_count, 1)
active_days = torch.unsqueeze(active_days, 1)
screen_name_length = torch.unsqueeze(screen_name_length, 1)
following_count = torch.unsqueeze(following_count, 1)
statues = torch.unsqueeze(statues, 1)

num_properties_tensor = torch.cat([followers_count,active_days,screen_name_length,following_count,statues],dim=1)
torch.save(num_properties_tensor, 'processed_data/num_properties.pt')
print(num_properties_tensor)

dict = {'followers_count_mean': float(followers_count_mean), 'followers_count_std': float(followers_count_std),
       'active_days_mean': float(active_days_mean), 'active_days_std': float(active_days_std),
       'screen_name_length_mean': float(screen_name_length_mean), 'screen_name_length_std': float(screen_name_length_std),
       'following_count_mean': float(following_count_mean), 'following_count_std': float(following_count_std),
       'statues_mean': float(statues_mean), 'statues_std': float(statues_std)}

np.save('processed_data/num_properties_meanstd.npy', dict)