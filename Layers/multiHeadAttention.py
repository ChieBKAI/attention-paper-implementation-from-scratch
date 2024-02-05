import torch
import torch.nn as nn
import math

# Multi-Head Attention
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model) 
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask=None, dropout: nn.Dropout = nn.Dropout(0.1)):
        d_k = q.size(-1)
        # (batch_size, h, seq_len, d_k) x (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # (batch_size, h, seq_len, seq_len) -> (batch_size, h, seq_len, d_k)
        attention_scores = torch.softmax(scores, dim=-1)
        attention_scores = dropout(attention_scores)

        # (batch_size, h, seq_len, d_k) x (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
        attention = torch.matmul(attention_scores, v)

        return attention, attention_scores

    def forward(self, q, k, v, mask=None):
        q = self.w_q(q) 
        k = self.w_k(k)
        v = self.w_v(v)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        q = q.view(q.size(0), q.size(1), self.h, self.d_k).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.h, self.d_k).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.h, self.d_k).transpose(1, 2)

        attention, self.attention_scores = self.attention(q, k, v, mask, self.dropout)
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        attention = attention.transpose(1, 2).contiguous().view(attention.size(0), -1, self.d_model)

        return self.w_o(attention) # (batch_size, seq_len, d_model)

