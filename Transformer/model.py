import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

class InputEmbedding(nn.Module):
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        
    # (batch, seq_len) -> (batch, seq_len, d_model)
    def forward(self, x):
        return self.token_embedding_table(x)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer('pos_table', self._positional_encoding())
        
    def _positional_encoding(self):
        pos_table = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(np.log(10000.0) / self.d_model))
        pos_table[:, 0::2] = torch.sin(position * div_term)
        pos_table[:, 1::2] = torch.cos(position * div_term)
        return pos_table
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        return x + self.pos_table[:seq_len, :].unsqueeze(0)
    

class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer: nn.Module):
        return x + self.dropout(sublayer(self.norm(x)))
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_head, d_model):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.n_head = n_head
        self.d_model = d_model
        self.head_size = d_model // n_head
        
        self.q = nn.ModuleList([nn.Linear(d_model, self.head_size) for _ in range(n_head)])
        self.k = nn.ModuleList([nn.Linear(d_model, self.head_size) for _ in range(n_head)])
        self.v = nn.ModuleList([nn.Linear(d_model, self.head_size) for _ in range(n_head)])
        
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=False):
        # (batch, seq_len, d_model) -> (batch, seq_len, head_size * n_head)
        Q = torch.cat([q(query) for q in self.q], dim=-1)
        K = torch.cat([k(key) for k in self.k], dim=-1)
        V = torch.cat([v(value) for v in self.v], dim=-1)
        
        batch, seq_len, _ = Q.shape
        
        # (batch, seq_len, head_size * n_head) -> 
        # (batch, n_head, seq_len, head_size) -> 
        Q = Q.view(batch, seq_len, self.n_head, -1).permute(0, 2, 1, 3)
        V = V.view(batch, seq_len, self.n_head, -1).permute(0, 2, 1, 3)
        K = K.view(batch, seq_len, self.n_head, -1).permute(0, 2, 1, 3)
        
        # (batch, n_head, seq_len, head_size) x (batch, n_head, head_size, seq_len) -> (batch, n_head, seq_len, seq_len)
        wei = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_size)
        
        if self.mask:
            mask = torch.tril(torch.ones(seq_len, seq_len))
            wei = wei.masked_fill(mask == 0, float('-inf'))
            
        # (batch, n_head, seq_len, seq_len) -> (batch, n_head, seq_len, seq_len)
        wei = torch.softmax(wei, dim=-1)
        
        # (batch, n_head, seq_len, seq_len) x (batch, n_head, seq_len, head_size) 
        # -> (batch, n_head, seq_len, head_size)
        out = wei @ V
        
        # (batch, n_head, seq_len, head_size) 
        # -> (batch, seq_len, n_head, head_size) 
        # -> (batch, seq_len, n_head * head_size)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, -1)
        
        out = self.out(out)
        
        return out

        
class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
    

class EncoderLayer(nn.Module):
    
    def __init__(self, 
                 features: int, 
                 self_attention_block: MultiHeadAttention, 
                 feed_forward_block: FeedForward, 
                 dropout: float):
        super().__init__()
        self.self_attention = self_attention_block
        self.feed_forward = feed_forward_block
        
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        
    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, mask=False))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    
    def __init__(self, featrues: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    

class DecoderLayer(nn.Module):

    def __init__(self,
                 features: int,
                 self_attention_block: MultiHeadAttention,
                 cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForward,
                 dropout: float):
        super().__init__()
        self.masked_self_attention = self_attention_block
        self.cross_attention = cross_attention_block
        self.feed_forward = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
        
    def forward(self, x, encoder_out):
        x = self.residual_connections[0](x, lambda x: self.masked_self_attention(x, x, x, mask=True))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_out, encoder_out, mask=False))
        x = self.residual_connections[2](x, self.feed_forward)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, d_model, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        
    def forward(self, x, encoder_out):
        for layer in self.layers:
            x = layer(x, encoder_out)
        
        return x
    
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size
        return self.fc(x)
    
class Transformer(nn.Module):
    
    def __init__(self, 
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbedding,
                 tgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer,
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection_layer
        
    def encode(self, src):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src)
    
    def decode(self, tgt, encoder_out):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_out)
    
    def forward(self, x, y):
        encoder_output = self.encode(x)
        decoder_output = self.decode(y, encoder_output)
        return self.projection(decoder_output)

    
def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int=512,
    N: int=6,
    h: int=8,
    dropout: float=0.1,
    d_ff: int=2048
    ) -> Transformer:
    
    src_embed = InputEmbedding(src_vocab_size, d_model)
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)
    
    src_pos = PositionalEncoding(d_model)
    tgt_pos = PositionalEncoding(d_model)
    
    encoder = Encoder(d_model, 
                      nn.ModuleList([
                            EncoderLayer(d_model, 
                                         MultiHeadAttention(h, d_model), 
                                         FeedForward(d_model, d_ff), dropout) for _ in range(N)
                            ]))
    
    decoder = Decoder(d_model, 
                      nn.ModuleList([
                          DecoderLayer(d_model, 
                                       MultiHeadAttention(h, d_model), 
                                       MultiHeadAttention(h, d_model), 
                                       FeedForward(d_model, d_ff), dropout) for _ in range(N)
                            ]))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer