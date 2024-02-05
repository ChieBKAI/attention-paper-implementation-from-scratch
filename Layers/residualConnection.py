import torch
import torch.nn as nn
from Layers.normalization import NormalizationLayer


# Residual Connection
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = NormalizationLayer()
    
    def forward(self, x, sublayer): # sublayer is a function
        return x + self.dropout(sublayer(self.norm(x))) # add but not norm