import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    # Later set in the build method according to the tokenizer
    vocab_size: int = -1 
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None
    

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float=10000.0):
    # Refer to the 3.2.2 of the paper
    # Rotary Position Embedding is only applied to Query and Key.
    assert head_dim % 2 == 0, "Dimension must be divisible by 2."
    
    # Step 1: Build the theta parameter
    # According to the formula of theta_i = theta^(-2(i-1)/dim) for i = [1, 2, ..., dim/2]
    # |Head_Dim / 2|
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # |Head_Dim / 2|
    theta = (1.0 / (theta ** (theta_numerator / head_dim))).to(device)
    
    # Step 2: Build the positions (m parameter)
    # |Seq_Len|
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position with outer product.
    # |Seq_Len, Head_Dim / 2|
    freqs = torch.outer(m, theta).float()
    # Constructs a complex tensor whose elements are Cartesian coordinates corresponding
    # to the polar coordinates with absolute value `abs` and angle `angle`.
    # = abs * cos(angle) + abs * sin(angle) * i
    # |Seq_Len, Head_Dim / 2|
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex
    

def apply_rotary_embeddings(x:torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # |B, Seq_Len, H, Head_Dim| -> |B, Seq_Len, H, Head_Dim / 2|
    # ..Example
    #   [[1, 2, 3, 4]] => [[1+2i], [3+4i]]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # |Seq_Len, Head_Dim / 2| -> |1, Seq_Len, 1, Head_Dim / 2|
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in x_complex tensor by the corresponding complex number in the freqs_complex,
    # which results in the rotation of the complex number.
    # Broad Casting Multiplication occurs
    # |B, Seq_Len, H, Head_Dim / 2| * |1, Seq_Len, 1, Head_Dim / 2| -> |B, Seq_Len, H, Head_Dim / 2|
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # |B, Seq_Len, H, Head_Dim / 2| -> |B, Seq_Len, H, Head_Dim / 2, 2|
    x_out = torch.view_as_real(x_rotated)
    # Reshape it
    x_out = x_out.reshape(*x.shape).type_as(x).to(device)
    return x_out

class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        # The gamma parameter in the paper
        self.weight = nn.Parameter(torch.ones(size=dim))
        
    def forward(self, x:torch.Tensor):
        # |B, Seq_Len, Dim| * |B, Seq_Len, 1| = |B, Seq_Len, Dim|
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // self.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # RMS normalization before the attention block
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        # RMS normalization before the feed forward block
        self.ffn_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # |B, Seq_Len, Dim|
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        # |B, Seq_Len, Dim|
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
        



    
class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super(Transformer, self).__init__()
        
        assert args.vocab_size != -1, "Vocab size must be set before initialization."
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            device=args.device
        )
            
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # |B, Seq_Len|; Seq_Len would be 1 due to KV cache
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed due to KV cache."
        
        # |B, Seq_Len| -> |B, Seq_Len, Dim|
        h = self.tok_embeddings(tokens)
        
        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos+seq_len]
        
        # Iteratively aply all the encoding layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        # apply RMS Normalization
        h = self.norm(h)
        
        # |B, Seq_Len, Vocab_Size|
        output = self.output(h)
        return output
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        