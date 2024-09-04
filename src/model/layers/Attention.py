import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self,embedding_size):
        super().__init__()
        self.W = nn.Linear(embedding_size, embedding_size, bias=False)
        self.U = nn.Linear(embedding_size, embedding_size, bias=False)
        self.V = nn.Linear(embedding_size, 1, bias=False)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(self, X, Q, mask=None, is_positive=False):
        X = self.W(X)
        Q = self.U(Q)
        bz = len(X)
        if len(Q.size()) < 3:
            length, dim = Q.size()
            Q = Q.unsqueeze(0)
            Q = Q.expand(bz, length, dim)
        att_weight = self.V(torch.tanh(X.unsqueeze(1) + Q)).squeeze(-1)
        if is_positive:
            att_weight = self.softplus(att_weight)
        if mask is not None:
            att_weight = torch.where(mask, torch.ones_like(att_weight) * -1e10, att_weight)
        att = torch.softmax(att_weight, dim=1)
        return att, att_weight
