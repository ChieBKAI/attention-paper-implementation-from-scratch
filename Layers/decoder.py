import torch.nn as nn
from Layers.decoderLayer import DecoderLayer
from Layers.normalization import NormalizationLayer

# Decoder
class Decoder(nn.Module):

    def __init__(self, d_model: int, h: int, d_ff: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)])
        self.norm = NormalizationLayer()
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x) # norm after residual connection layer