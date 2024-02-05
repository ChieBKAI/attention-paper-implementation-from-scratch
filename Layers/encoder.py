import torch.nn as nn
from Layers.encoderLayer import EncoderLayer
from Layers.normalization import NormalizationLayer


# Encoder
class Encoder(nn.Module):

    def __init__(self, d_model: int, h: int, d_ff: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)])
        self.norm = NormalizationLayer()
    
    def forward(self, x, mask=None):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return self.norm(x) # norm after residual connection layer