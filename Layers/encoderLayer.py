import torch.nn as nn
from Layers.multiHeadAttention import MultiHeadAttention
from Layers.feedForwardNetwork import FeedForwardNetwork
from Layers.residualConnection import ResidualConnection


# Encoder Layer
class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.self_attention_layer = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward_network = FeedForwardNetwork(d_model, d_ff, dropout)
        self.residual_connection_1 = ResidualConnection(dropout)
        self.residual_connection_2 = ResidualConnection(dropout)
    
    def forward(self, x, src_mask=None):
        x = self.residual_connection_1(x, lambda x: self.self_attention_layer(x, x, x, src_mask))
        x = self.residual_connection_2(x, self.feed_forward_network)
        return x # (batch_size, seq_len, d_model)