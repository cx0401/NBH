import time
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Task2_2208_2212_config, Task2_2212_2304_config
from utils import date, evaluate_mse, MFDataset, evaluate_top_n, evaluate_precision, split_dataset, evaluate_predict
from model import FunkSVD
import json

def main():
    task = sys.argv[1]
    configs = {
        "task2_2208_2212": Task2_2208_2212_config(),
        "task2_2212_2304": Task2_2212_2304_config()
    }
    config = configs[task]
    print(config)

    df = pd.read_csv(config.dataset_file, usecols=['position_code', 'skill_id', 'skill_score'])
    df.columns = ['position_id', 'skill_id', 'rating']  # Rename above columns for convenience
    # map user(or item) to number
    df['userID'] = df.groupby(df['position_id']).ngroup()
    df['itemID'] = df.groupby(df['skill_id']).ngroup()
    user_count = df['userID'].value_counts().count()  # 用户数量
    item_count = df['itemID'].value_counts().count()  # item数量
    print(f"{date()}## Dataset contains {df.shape[0]} records, {user_count} users and {item_count} items.")

    # model = FunkSVD(user_count, item_count, config.embedding_dim).to(config.device)
    model = torch.load(config.saved_model)

    pos = model.user_emb
    skill = model.item_emb

    mp = {df['userID'][i]: str(df['position_id'][i]) for i in range(len(df['userID']))}

    similarity = torch.cosine_similarity(pos.unsqueeze(1), pos.unsqueeze(0), dim=-1)
    position_config = json.load(open(config.position_config_file,"r"))

    ans = {}
    for i in range(similarity.shape[0]):
        ans[position_config[mp[i]]] = {}
        for j in range(similarity.shape[1]):
            ans[position_config[mp[i]]][position_config[mp[j]]] = similarity[i][j].item()

    pd.DataFrame(ans).to_csv(config.similarity_file)


if __name__ == '__main__':
    main()
