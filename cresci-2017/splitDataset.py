import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 读取原始的label.csv文件
df = pd.read_csv('../datasets/cresci-2015/label.csv')

# 按照标签分组
human_df = df[df['label'] == 'human']
bot_df = df[df['label'] == 'bot']

# 确定分组后的数量
half_human = len(human_df) // 2
half_bot = len(bot_df) // 2

# 分割数据集
label_attack_human = human_df.iloc[:half_human]
label_defense_human = human_df.iloc[half_human:]

label_attack_bot = bot_df.iloc[:half_bot]
label_defense_bot = bot_df.iloc[half_bot:]

# 合并human和bot
label_attack = pd.concat([label_attack_human, label_attack_bot])
label_defense = pd.concat([label_defense_human, label_defense_bot])

# 打乱数据
label_attack = label_attack.sample(frac=1).reset_index(drop=True)
label_defense = label_defense.sample(frac=1).reset_index(drop=True)

# 打印两个文件的bot和human数量
print("label_attack.csv 中的数量分布:")
print(label_attack['label'].value_counts())

print("label_defense.csv 中的数量分布:")
print(label_defense['label'].value_counts())

# # 保存到新的CSV文件
# label_attack.to_csv('../datasets/cresci-2015/label_attack.csv', index=False)
# label_defense.to_csv('../datasets/cresci-2015/label_defense.csv', index=False)

# 读取原始数据
df = pd.read_csv('../datasets/cresci-2015/label_defense.csv')

# 按比例分割数据集
train, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

# 添加split列
train['split'] = 'train'
val['split'] = 'val'
test['split'] = 'test'
# 合并所有数据
split_df = pd.concat([train, val, test])
# 只保留id和split列
split_df = split_df[['id', 'split']]
# 输出到split_attack.csv
split_df.to_csv('../datasets/cresci-2015/split_defense.csv', index=False)
# 打印结果以检查
print(split_df['split'].value_counts())

print("分割完成并保存为label_attack.csv和label_defense.csv")
