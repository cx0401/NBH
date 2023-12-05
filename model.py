import torch
import torch.nn as nn

class FunkSVD(nn.Module):

    def __init__(self, M, N, K=10):
        super().__init__()
        self.user_emb = nn.Parameter(torch.randn(M, K))
        self.user_bias = nn.Parameter(torch.randn(M)) 
        self.item_emb = nn.Parameter(torch.randn(N, K))
        self.item_bias = nn.Parameter(torch.randn(N))
        self.bias = nn.Parameter(torch.zeros(1))  

    def forward(self, user_id, item_id):
        pred = (self.user_emb[user_id] + self.user_bias[user_id]) * (self.item_emb[item_id] + self.item_bias[item_id])+ self.bias
        # pred = pred.sum(dim=-1) + self.user_bias[user_id] + self.item_bias[item_id] + self.bias
        return pred