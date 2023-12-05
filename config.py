import argparse
import inspect

import torch

class Config():
    train_epochs = 50
    batch_size = 1024
    learning_rate = 0.01
    l2_regularization = 1e-3  
    learning_rate_decay = 0.99  
    embedding_dim = 10
    device = torch.device("cuda:0")

class Task1_config(Config):
    dataset_file = 'data/task1/pos_skill_score.csv'
    saved_model = 'models/best_model.pt'
    result_file = 'results/gpt_impact.csv'
    position_config_file = 'data/task1/position_config.csv'

class Task2_2208_2212_config(Config):
    dataset_file = 'data/task2/gpt_2208_2212_skill_relation.csv'
    saved_model = 'models/2208_2212_model.pt'
    result_file = 'results/gpt_impact_2208_2212.csv'
    similarity_file = 'results/similarity_2208_2212.csv'
    position_config_file = 'data/task2/gpt_2208_2212_position_code_name.json'

class Task2_2212_2304_config(Config):
    dataset_file = 'data/task2/gpt_2212_2304_skill_relation.csv'
    saved_model = 'models/2212_2304_model.pt'
    result_file = 'results/gpt_impact_2212_2304.csv'
    similarity_file = 'results/similarity_2212_2304.csv'
    position_config_file = 'data/task2/gpt_2212_2304_position_code_name.json'

class Task3_config(Config):
    dataset_file = f'data/task3/Onet_triplet.tsv'
    saved_model = f'models/task3_best_model.pt'
    gpt_score_file = f'data/task3/ONET_annotation_0821.xlsx'
    catory = 'AVERAGE'
    # catory = 'GPT_AVG'
    # catory = 'PEO_AVG'
    result_file = f'results/task3_gpt_impact_{catory}.csv'
