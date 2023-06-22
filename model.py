import torch
import torch.nn as nn

class FunkSVD(nn.Module):

    def __init__(self, M, N, K=10):
        super().__init__()
        self.user_emb = nn.Parameter(torch.randn(M, K))
        self.user_bias = nn.Parameter(torch.randn(M))  # 偏置
        self.item_emb = nn.Parameter(torch.randn(N, K))
        self.item_bias = nn.Parameter(torch.randn(N))
        self.bias = nn.Parameter(torch.zeros(1))  # 全局偏置

    def forward(self, user_id, item_id):
        pred = self.user_emb[user_id] * self.item_emb[item_id]
        pred = pred.sum(dim=-1) + self.user_bias[user_id] + self.item_bias[item_id] + self.bias
        return pred