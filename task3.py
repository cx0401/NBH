import time
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Task4_config
from utils import date, evaluate_mse, MFDataset, evaluate_top_n, evaluate_precision, split_dataset, evaluate_predict
from model import FunkSVD

def main():
    config = Task4_config()
    print(config)

    relation = pd.read_excel(config.gpt_score_file)

    df = pd.read_csv(config.dataset_file, usecols=['position_code', 'skill_id', 'skill_score'])
    df.columns = ['user_name', 'item_name', 'rating'] 
    df['userID'] = df.groupby(df['user_name']).ngroup()
    df['itemID'] = df.groupby(df['item_name']).ngroup()
    user_count = df['userID'].value_counts().count() 
    item_count = df['itemID'].value_counts().count()  
    print(f"{date()}## Dataset contains {df.shape[0]} records, {user_count} users and {item_count} items.")

    dataset = MFDataset(df)
    dlr = DataLoader(dataset, batch_size=config.batch_size)

    model = torch.load(config.saved_model)

    pos = model.user_emb
    skill = model.item_emb

    mp = {df['item_name'][i]:df['itemID'][i] for i in range(len(df['itemID']))}
    item_ids = [mp[i] for i in relation['item_name'] if i in mp]
    related = torch.tensor([relation[config.catory][index] for index, i in enumerate(relation['item_name'])  if i in mp], dtype=torch.float32).unsqueeze(0).to('cuda')

    gpt_embedding = torch.matmul(related, model.item_emb[item_ids])
    gpt_bias = torch.matmul(related, model.item_bias[item_ids]) 

    pred = model.user_emb * gpt_embedding 
    result = pred.sum(dim=-1) + model.user_bias + gpt_bias + model.bias
    result = result / related.abs().sum()

    mp_position = {df['userID'][i]:df['user_name'][i] for i, _ in enumerate(df['userID'])}
    position_names = [mp_position[i] for i, user_id in enumerate(mp_position)]

    impact_gpt = {
        "name": [],
        "score": [],
        "var": []
    }

    pred_matrix = torch.matmul(model.user_emb, model.item_emb.T) + model.item_bias 
    pred_matrix_gpt = pred_matrix * related
    means = torch.mean(pred_matrix_gpt, dim=-1)
    vars = torch.var(pred_matrix_gpt, dim=-1)

    for i in range(len(position_names)):
        impact_gpt['score'].append(result[i].item())
        impact_gpt['var'].append(vars[i].item())
        impact_gpt['name'].append(position_names[i])
    
    pd.DataFrame(impact_gpt).to_csv(config.result_file)

if __name__ == '__main__':
    main()
