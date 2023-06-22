import time
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Task1_config
from utils import date, evaluate_mse, MFDataset, evaluate_top_n, evaluate_precision, split_dataset, evaluate_predict
from model import FunkSVD

def main():
    config = Task1_config()
    print(config)

    relation = pd.read_excel('data/task1/gpt_score.xlsx')

    df = pd.read_csv(config.dataset_file, usecols=['position_code', 'skill_id', 'skill_score'])
    df.columns = ['position_id', 'skill_id', 'rating']  # Rename above columns for convenience
    # map user(or item) to number
    df['userID'] = df.groupby(df['position_id']).ngroup()
    df['itemID'] = df.groupby(df['skill_id']).ngroup()
    user_count = df['userID'].value_counts().count()  # 用户数量
    item_count = df['itemID'].value_counts().count()  # item数量
    print(f"{date()}## Dataset contains {df.shape[0]} records, {user_count} users and {item_count} items.")

    dataset = MFDataset(df)
    dlr = DataLoader(dataset, batch_size=config.batch_size)

    # model = FunkSVD(user_count, item_count, config.embedding_dim).to(config.device)
    model = torch.load(config.saved_model)

    pos = model.user_emb
    skill = model.item_emb

    mp = {df['skill_id'][i]:df['itemID'][i] for i in range(len(df['itemID']))}
    item_ids = [mp[i] for i in relation['skill_id'] if i in mp]
    related = torch.tensor([relation['Average'][index] for index, i in enumerate(relation['skill_id'])  if i in mp], dtype=torch.float32).unsqueeze(0).to('cuda')

    gpt_embedding = torch.matmul(related, model.item_emb[item_ids])
    gpt_bias = torch.matmul(related, model.item_bias[item_ids]) 

    pred = model.user_emb * gpt_embedding 
    result = pred.sum(dim=-1) + model.user_bias + gpt_bias + model.bias
    result = result / related.abs().sum()
    # result = result / related.sum()


    mp_position = {df['userID'][i]:df['position_id'][i] for i, _ in enumerate(df['userID'])}
    position_codes = [mp_position[i] for i, user_id in enumerate(mp_position)]
    position_config = pd.read_csv(config.position_config_file, sep=',')
    position_code_name = {code:position_config['position_name'][i] for i, code in enumerate(position_config['position_code'])}

    impact_gpt = {
        "code": [],
        "name": [],
        "score": [],
        "var": []
    }

    pred_matrix = torch.matmul(model.user_emb, model.item_emb.T) + model.item_bias 
    pred_matrix_gpt = pred_matrix * related
    means = torch.mean(pred_matrix_gpt, dim=-1)
    vars = torch.var(pred_matrix_gpt, dim=-1)

    for i in range(len(position_codes)):
        impact_gpt['score'].append(result[i].item())
        impact_gpt['var'].append(vars[i].item())
        impact_gpt['name'].append(position_code_name[position_codes[i]])
        impact_gpt['code'].append(position_codes[i])
    
    pd.DataFrame(impact_gpt).to_csv(config.result_file)

if __name__ == '__main__':
    main()
