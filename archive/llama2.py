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
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    
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

def repeat_kv(x:torch.Tensor, n_rep:int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # |B, Seq_Len, H_KV, 1, Head_Dim|
            x[:, :, :, None, :]
            # |B, Seq_Len, H_KV, N_Rep, Head_Dim|
            # .repeat(1, 1, 1, n_rep, 1)
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            # |B, Seq_Len, H_KV * N_Rep, Head_Dim|
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        # The `gamma` parameter in the paper
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x:torch.Tensor):
        # |B, Seq_Len, Dim| * |B, Seq_Len, 1| = |B, Seq_Len, Dim|
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # |Dim| * |B, Seq_Len, Dim|
        return self.weight * self._norm(x.float()).type_as(x)

class SelfAttention(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super(SelfAttention, self).__init__()
        
        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor, mask: Optional[torch.Tensor],):
        batch_size, seq_len, _ = x.shape # |B, 1, Dim|
        
        # |B, 1, Dim| -> |B, 1, H_Q * Head_Dim|
        xq = self.wq(x)
        # |B, 1, Dim| -> |B, 1, H_KV * Head_Dim|
        xk = self.wk(x)
        # |B, 1, Dim| -> |B, 1, H_VV * Head_Dim|
        xv = self.wv(x)
        
        # |B, 1, H_Q * Head_Dim| -> |B, 1, H_Q, Head_Dim|
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # |B, 1, H_KV * Head_Dim| -> |B, 1, H_KV, Head_Dim|
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # |B, 1, H_KV * Head_Dim| -> |B, 1, H_KV, Head_Dim|
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # |B, 1, H_Q, Head_Dim| -> |B, 1, H_Q, Head_Dim|
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # |B, 1, H_KV, Head_Dim| -> |B, 1, H_KV, Head_Dim|
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        
        # Fill the entry in cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv
        
        # |B, Seq_Len_KV, H_KV, Head_Dim|
        keys = self.cache_k[:batch_size, :start_pos+seq_len]
        values = self.cache_v[:batch_size, :start_pos+seq_len]
        
        # Since every group of Q shares the same K and V heads, just repeat (copy) the K and V heads for every Q in the same group.
        # |B, Seq_Len_KV, H_KV, Head_Dim| -> |B, Seq_Len_KV, H_Q, Head_Dim|
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        # |B, 1, H_Q, Head_Dim| -> # |B, H_Q, 1, Head_Dim|
        xq = xq.transpose(1, 2)
         # |B, Seq_Len_KV, H_Q, Head_Dim| -> |B, H_Q, Seq_Len_KV, Head_Dim|
        keys = keys.transpose(1, 2)
        # |B, Seq_Len_KV, H_Q, Head_Dim| -> |B, H_Q, Seq_Len_KV, Head_Dim|
        values = values.transpose(1, 2)
        
        # |B, H_Q, 1, Head_Dim| @ |B, H_Q, Head_Dim, Seq_Len_KV| -> |B, H_Q, 1, Seq_Len_KV|
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            # In case the sequence length is not 0
            # |B, H_Q, Seq_Len, Seq_Len_KV| + |Seq_Len, Seq_Len_KV + Seq_Len| -> |B, H_Q, Seq_Len, Seq_Len_KV|
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # |B, H_Q, 1, Seq_Len_KV| @ |B, H_Q, Seq_Len_KV, Head_Dim}| -> |B, H_Q, 1, Head_Dim|
        output = torch.matmul(scores, values)
        # |B, H_Q, 1, Head_Dim| -> |B, 1, H_Q, Head_Dim| -> |B, 1, Dim|
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        # |B, 1, Dim|
        return self.wo(output)
    
    
class FeedForward(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super(FeedForward, self).__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest `multiple_of` parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor):
        # |B, Seq_Len, Dim| -> |B, Seq_Len, Hidden_Dim|
        swish = F.silu(self.w1(x))
        # |B, Seq_Len, Dim| -> |B, Seq_Len, Hidden_Dim|
        x_V = self.w3(x)
        # |B, Seq_Len, Hidden_Dim| * |B, Seq_Len, Hidden_Dim| -> |B, Seq_Len, Hidden_Dim|
        x = swish * x_V
        # |B, Seq_Len, Hidden_Dim| -> |B, Seq_Len, Dim|
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super(EncoderBlock, self).__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // self.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # RMS normalization before the attention block
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        # RMS normalization before the feed forward block
        self.ffn_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # |B, Seq_Len, Dim|
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex, mask
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
    
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # |B, Seq_Len|; Seq_Len would be 1 due to KV cache
        # However, it also could be larger than 1. In this case, we have to input mask.
        batch_size, seq_len = tokens.shape
        
        # |B, Seq_Len| -> |B, Seq_Len, Dim|
        h = self.tok_embeddings(tokens)
        
        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos+seq_len]
        
        mask = None
        if seq_len > 1:
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            
            # Horizontal stack
            # |Seq_Len, Seq_Len_KV| & |Seq_Len, Seq_Len| -> |Seq_Len, Seq_Len_KV + Seq_Len|
            mask = torch.hstack([
                torch.zeros((seq_len, start_pos), device=tokens.device),
                mask
            ]).type_as(h)
        
        # Iteratively aply all the encoding layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex, mask=mask)
        
        # apply RMS Normalization
        h = self.norm(h)
        
        # |B, Seq_Len, Vocab_Size|
        output = self.output(h).float()
        return output
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        