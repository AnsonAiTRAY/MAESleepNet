import torch
from torch import nn
from torch.nn import functional as F
import math


class AttentionMix(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMix, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x is of shape [batch_size, seq_len, input_dim]
        att_scores = self.linear2(torch.tanh(self.linear1(x)))
        att_weights = F.softmax(att_scores.squeeze(-1), dim=-1)
        weighted_x = x * att_weights.unsqueeze(-1)
        output = weighted_x.sum(dim=1)
        return output


class MultiEpochClassifier(nn.Module):
    def __init__(self, num_token, embed_dim, pre_model, clip_num, length, num_class=5):
        super().__init__()
        self.num_token = num_token
        self.embed_dim = embed_dim
        self.clip_num = clip_num
        self.num_class = num_class
        self.length = length
        self.MAE = pre_model
        self.gru = nn.GRU(input_size=self.num_token * self.embed_dim, hidden_size=1024, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.attn_mix = AttentionMix(self.num_token * self.embed_dim)
        self.linear = nn.Sequential(nn.Linear(2048, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, num_class))

    def forward(self, x):
        hidden1 = self.MAE(x)
        # return hidden1
        hidden = hidden1.view(hidden1.size(0) // self.clip_num, hidden1.size(1) * self.clip_num, -1)
        out = self.attn_mix(hidden)
        out = out.reshape(out.shape[0] // self.length, self.length, -1)
        out, _ = self.gru(out, None)
        # return out
        pred = self.linear(out)
        return pred
