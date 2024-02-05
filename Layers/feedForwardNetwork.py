import torch
import torch.nn as nn

# Feed Forward Network
class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.ff1(x)))
        return self.ff2(x)
