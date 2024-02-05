import torch
import torch.nn as nn
import math

# Embedding
class Embedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
    
    def forward(self, x):
        return self.embedding(x) * self.scale 