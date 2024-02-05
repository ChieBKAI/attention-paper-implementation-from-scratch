import torch
import torch.nn as nn
from Layers.encoder import Encoder
from Layers.decoder import Decoder
from Layers.embedding import Embedding
from Layers.positionalEncoding import PositionalEncoding
from Layers.projectionLayer import ProjectionLayer
from Layers.encoderLayer import EncoderLayer
from Layers.decoderLayer import DecoderLayer
from Layers.transformer import Transformer

# Build Transformer model
def build_model(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, h: int = 8, d_ff: int = 2048, num_layers: int = 6, dropout: float = 0.1) -> Transformer:
    src_embedding = Embedding(src_vocab_size, d_model)
    tgt_embedding = Embedding(tgt_vocab_size, d_model)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create Encoder and Decoder Layers
    encoder_blocks = []
    for _ in range(num_layers):
        encoder_blocks.append(EncoderLayer(d_model, h, d_ff, dropout))
    decoder_blocks = []
    for _ in range(num_layers):
        decoder_blocks.append(DecoderLayer(d_model, h, d_ff, dropout))

    # Create Encoder and Decoder
    encoder = Encoder(d_model, h, d_ff, num_layers, dropout)
    decoder = Decoder(d_model, h, d_ff, num_layers, dropout)

    # Create Projection Layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create Transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection_layer)

    # Initialize parameters with Glorot / fan_avg.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer