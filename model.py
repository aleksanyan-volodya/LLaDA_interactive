"""
Bidirectional mask predictor

The model only sees the partially masked sequence and returns
logits over the vocabulary for every position.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8000
    seq_len: int = 256
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mask_id: int = 1


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_tables(seq_len, head_dim, base=10000.0):
    freqs = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(seq_len).float(), freqs)  # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, head_dim). Rotate the two halves of each head as pairs.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # bidirectional
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = 64 * round(8 * cfg.dim / 3 / 64)
        self.gate = nn.Linear(cfg.dim, hidden, bias=False)
        self.up = nn.Linear(cfg.dim, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class MaskPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg.seq_len, cfg.dim // cfg.n_heads)
        self.register_buffer("cos", cos, persistent=False)  # not saved
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        # idx: (B, T) token ids, some equal to cfg.mask_id. Returns logits (B, T, vocab_size).
        T = idx.shape[1]
        cos, sin = self.cos[:T], self.sin[:T]
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.norm(x))


if __name__ == "__main__":
    full = MaskPredictor(Config())
    print(f"full config: {sum(p.numel() for p in full.parameters()) / 1e6:.1f}M parameters")

    cfg = Config(dim=64, n_layers=2, n_heads=4, seq_len=32)
    model = MaskPredictor(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    print("tiny config: logits shape", tuple(model(idx).shape))
