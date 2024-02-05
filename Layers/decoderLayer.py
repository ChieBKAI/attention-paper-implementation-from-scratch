import torch.nn as nn
from Layers.multiHeadAttention import MultiHeadAttention
from Layers.feedForwardNetwork import FeedForwardNetwork
from Layers.residualConnection import ResidualConnection

# Decoder Layer
class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.self_attention_layer = MultiHeadAttention(d_model, h, dropout)
        self.cross_attention_layer = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward_network = FeedForwardNetwork(d_model, d_ff, dropout)
        self.residual_connection_1 = ResidualConnection(dropout)
        self.residual_connection_2 = ResidualConnection(dropout)
        self.residual_connection_3 = ResidualConnection(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.residual_connection_1(x, lambda x: self.self_attention_layer(x, x, x, tgt_mask))
        x = self.residual_connection_2(x, lambda x: self.cross_attention_layer(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection_3(x, self.feed_forward_network)
        return x # (batch_size, seq_len, d_model)