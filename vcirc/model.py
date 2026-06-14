"""TinyTransformer: a minimal, transparent transformer for finite-domain tasks.

Design choices that matter for later verification:
  - causal attention  -> each position can compute its running prefix statistic
  - >=2 layers        -> compose per-position features (e.g. depth -> violations)
  - sum-pool          -> aggregate per-position features (counts/sums) exactly
  - MLP readout head  -> form non-monotone tests (e.g. |final depth| == 0)
  - NO LayerNorm      -> keeps later exact / rational re-evaluation clean
"""

import math

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.heads, self.d = heads, d
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.mlp1 = nn.Linear(d, ff)
        self.mlp2 = nn.Linear(ff, d)

    def forward(self, z: torch.Tensor, causal: bool) -> torch.Tensor:
        B, n, d = z.shape
        h, dh = self.heads, d // self.heads
        q = self.Wq(z).view(B, n, h, dh).transpose(1, 2)
        k = self.Wk(z).view(B, n, h, dh).transpose(1, 2)
        v = self.Wv(z).view(B, n, h, dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        if causal:
            m = torch.triu(torch.ones(n, n, device=z.device), diagonal=1).bool()
            att = att.masked_fill(m, float("-inf"))
        att = att.softmax(-1)
        ctx = (att @ v).transpose(1, 2).reshape(B, n, d)
        z = z + self.Wo(ctx)
        z = z + self.mlp2(torch.relu(self.mlp1(z)))
        return z


class TinyTransformer(nn.Module):
    def __init__(self, n: int, d: int = 32, heads: int = 2, ff: int = 64,
                 layers: int = 2, causal: bool = True, pool: str = "sum"):
        super().__init__()
        self.cfg = dict(n=n, d=d, heads=heads, ff=ff, layers=layers,
                        causal=causal, pool=pool)
        self.n, self.causal, self.pool = n, causal, pool
        self.tok = nn.Embedding(2, d)
        self.pos = nn.Parameter(torch.randn(n, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.h1 = nn.Linear(d, d)
        self.h2 = nn.Linear(d, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.tok(x) + self.pos[None, :, :]
        for b in self.blocks:
            z = b(z, self.causal)
        pooled = z.sum(1) if self.pool == "sum" else z.mean(1)
        return self.h2(torch.relu(self.h1(pooled)))
