import torch.nn as nn
from torch.nn import functional as F
import torch

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size() #x.shape B,T,C
        qkv = self.c_attn(x) # B,T, C*3
        q, k, v = qkv.split(self.n_embd, dim = 2)
        
        q = q.view(B,T, self.n_head, C // self.n_head).transpose(1, 2) # B, n_head, T, C
        k = k.view(B,T, self.n_head, C // self.n_head).transpose(1, 2) # B, n_head, T, C
        v = v.view(B,T, self.n_head, C // self.n_head).transpose(1, 2) # B, n_head, T, C
        # attn_weights = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(q.shape[-1]))
        # attn_weights = attn_weights.masked_fill(self.bias[:,:, :T, :T] == 0, float('-inf'))
        # attn_weights = F.softmax(attn_weights, dim = -1)

        # attn_out = attn_weights @ v
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        
        attn_out = attn_out.transpose(1,2).contiguous().view(B,T,C)
        proj_out = self.c_proj(attn_out)
        return proj_out
        