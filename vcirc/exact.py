"""Rigorous interval evaluation of the TinyTransformer — the v2 certificate core.

This re-implements `vcirc.model.TinyTransformer.forward` in **rigorous interval
arithmetic** so we can *prove* (not just observe) that the decision margin is
strictly positive on every input.

Why this is tractable (see docs/design.md): the weights are exact dyadic
rationals (float32), the inputs are exact integers, and the attention scale
`1/sqrt(dh) = 1/4` is exact. The *only* transcendental in the decision path is
`exp` inside the two attention softmaxes — and the argmax decision does not
involve the output softmax at all. After the first softmax the computation is
interval-valued (layer-2 Q/K/V, the second `exp` of interval arguments, the
readout), so everything downstream is interval arithmetic with rational
endpoints.

Representation. Every interval endpoint is an integer in units of ``2**-P``
(``P = PRECISION`` bits): the value ``v`` is stored as the integer
``round(v * 2**P)``. All operations round **outward** (lower endpoints toward
-inf, upper endpoints toward +inf) using exact integer floor/ceil division, so
the result is always a rigorous enclosure of the true real value — just with a
bounded denominator (``2**P``) that keeps numerators from ballooning through the
two layers. Increasing ``P`` only tightens the (already ~1e-21) width.

This is a *verified enclosure* of the exact-real function defined by the weights;
the margin lower bound it produces is a rigorous rational lower bound. Pure
standard library — integers and `fractions.Fraction` only (no torch, no numpy).
This module is the trusted core shared by the generator (`vcirc.certify`) and the
standalone checker (`certificates/check.py`).
"""

from __future__ import annotations

import json
import multiprocessing as mp
from fractions import Fraction
from math import isqrt
from typing import List, Optional, Sequence

PRECISION: int = 96          # bits after the binary point (dyadic precision)
_SCALE: int = 1 << PRECISION


def set_precision(p: int) -> None:
    """Set the dyadic precision (bits) used for outward rounding."""
    global PRECISION, _SCALE
    PRECISION = p
    _SCALE = 1 << p


def _ceil_div(a: int, b: int) -> int:
    """Ceil(a / b) for integer a and positive integer b."""
    return -((-a) // b)


class Ival:
    """A rigorous interval whose endpoints are integers in units of 2**-P.

    ``lo_i / 2**P <= true value <= hi_i / 2**P``.
    """

    __slots__ = ("lo_i", "hi_i")

    def __init__(self, lo_i: int, hi_i: int):
        self.lo_i = lo_i
        self.hi_i = hi_i

    @classmethod
    def point(cls, v: Fraction) -> "Ival":
        """Tightest interval around an exact rational (a weight/embedding)."""
        num, den = v.numerator, v.denominator
        lo = (num * _SCALE) // den
        hi = _ceil_div(num * _SCALE, den)
        return cls(lo, hi)

    def __add__(self, o: "Ival") -> "Ival":
        return Ival(self.lo_i + o.lo_i, self.hi_i + o.hi_i)

    def __sub__(self, o: "Ival") -> "Ival":
        return Ival(self.lo_i - o.hi_i, self.hi_i - o.lo_i)

    def __neg__(self) -> "Ival":
        return Ival(-self.hi_i, -self.lo_i)

    def mul_scalar(self, w: Fraction) -> "Ival":
        """Multiply by an exact rational scalar (a weight), rounding outward."""
        wn, wd = w.numerator, w.denominator     # wd > 0
        if wn >= 0:
            return Ival((self.lo_i * wn) // wd, _ceil_div(self.hi_i * wn, wd))
        return Ival((self.hi_i * wn) // wd, _ceil_div(self.lo_i * wn, wd))

    def __mul__(self, o: "Ival") -> "Ival":
        a, b, c, d = self.lo_i, self.hi_i, o.lo_i, o.hi_i
        p1, p2, p3, p4 = a * c, a * d, b * c, b * d   # units 2**-2P
        lo = min(p1, p2, p3, p4) >> PRECISION          # floor / 2**P
        hi = -((-max(p1, p2, p3, p4)) >> PRECISION)    # ceil  / 2**P
        return Ival(lo, hi)

    def relu(self) -> "Ival":
        return Ival(max(0, self.lo_i), max(0, self.hi_i))

    @property
    def lo(self) -> Fraction:
        return Fraction(self.lo_i, _SCALE)

    @property
    def hi(self) -> Fraction:
        return Fraction(self.hi_i, _SCALE)

    def width(self) -> Fraction:
        return Fraction(self.hi_i - self.lo_i, _SCALE)

    def max_bits(self) -> int:
        return max(self.lo_i.bit_length(), self.hi_i.bit_length())


# --------------------------------------------------------------------------- #
# rigorous exp enclosure (fixed-point), for an argument given in units of 2**-P
# --------------------------------------------------------------------------- #
def _exp_lower_nonneg(X: int) -> int:
    """Lower bound on exp(x), x = X/2**P >= 0; every term rounded DOWN."""
    N = (X >> PRECISION) + 60
    term = _SCALE
    S = _SCALE
    for k in range(1, N + 1):
        term = (term * X) // (k * _SCALE)     # x^k/k!, floored
        if term == 0:
            break                             # omitted terms are >= 0
        S += term
    return S


def _exp_upper_nonneg(X: int) -> int:
    """Upper bound on exp(x), x = X/2**P >= 0; terms rounded UP + tail bound."""
    N = (X >> PRECISION) + 60
    term = _SCALE
    S = _SCALE
    for k in range(1, N + 1):
        term = _ceil_div(term * X, k * _SCALE)
        S += term
    next_term = _ceil_div(term * X, (N + 1) * _SCALE)
    ratio_up = _ceil_div(X, N + 2)            # x/(N+2) in units 2**-P, ceil
    denom = _SCALE - ratio_up                 # (1 - ratio) in units 2**-P
    assert denom > 0, "exp tail ratio >= 1; increase term count"
    tail = _ceil_div(next_term * _SCALE, denom)
    return S + tail


def exp_bounds(X: int):
    """Rigorous [lo_i, hi_i] (units 2**-P) enclosing exp(X/2**P) for any int X."""
    if X >= 0:
        return _exp_lower_nonneg(X), _exp_upper_nonneg(X)
    lo_p = _exp_lower_nonneg(-X)              # 0 < lo_p <= exp(-x)*2**P <= hi_p
    hi_p = _exp_upper_nonneg(-X)
    return (_SCALE * _SCALE) // hi_p, _ceil_div(_SCALE * _SCALE, lo_p)   # 1/exp(-x)


def exp_ival(a: Ival) -> Ival:
    """Enclose exp over an interval (exp is monotone increasing)."""
    lo, _ = exp_bounds(a.lo_i)
    _, hi = exp_bounds(a.hi_i)
    return Ival(lo, hi)


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def _F(v) -> Fraction:
    """Exact Fraction from a float (dyadic), a [num, den] pair, or a Fraction."""
    if isinstance(v, Fraction):
        return v
    if isinstance(v, (list, tuple)):
        return Fraction(int(v[0]), int(v[1]))
    return Fraction(v)  # float -> exact dyadic rational


def _mat(rows) -> List[List[Fraction]]:
    return [[_F(x) for x in row] for row in rows]


def _vec(xs) -> List[Fraction]:
    return [_F(x) for x in xs]


class Weights:
    """Exact-rational weights of a TinyTransformer, grouped for the forward pass.

    Built from nested Python lists of floats (torch ``.tolist()``) or of
    ``[num, den]`` pairs / Fractions — all converted to exact `Fraction`.
    """

    def __init__(self, cfg: dict, sd: dict):
        self.cfg = dict(cfg)
        self.n = cfg["n"]; self.d = cfg["d"]; self.heads = cfg["heads"]
        self.ff = cfg["ff"]; self.layers = cfg["layers"]
        self.dh = self.d // self.heads
        r = isqrt(self.dh)
        assert r * r == self.dh, f"sqrt(dh)={self.dh} is not exact; scale not rational"
        self.scale = Fraction(1, r)
        self.tok = _mat(sd["tok.weight"])              # (2, d)
        self.pos = _mat(sd["pos"])                     # (n, d)
        self.blocks = []
        for i in range(self.layers):
            p = f"blocks.{i}."
            self.blocks.append(dict(
                Wq=_mat(sd[p + "Wq.weight"]), Wk=_mat(sd[p + "Wk.weight"]),
                Wv=_mat(sd[p + "Wv.weight"]), Wo=_mat(sd[p + "Wo.weight"]),
                mlp1_w=_mat(sd[p + "mlp1.weight"]), mlp1_b=_vec(sd[p + "mlp1.bias"]),
                mlp2_w=_mat(sd[p + "mlp2.weight"]), mlp2_b=_vec(sd[p + "mlp2.bias"]),
            ))
        self.h1_w = _mat(sd["h1.weight"]); self.h1_b = _vec(sd["h1.bias"])
        self.h2_w = _mat(sd["h2.weight"]); self.h2_b = _vec(sd["h2.bias"])

    @classmethod
    def from_hex_export(cls, path: str) -> "Weights":
        """Load weights from a torch-free hex export (see vcirc/export_weights.py).

        Each value is stored as `float.hex()`, which round-trips the float32
        weight exactly — so the checker reconstructs the *exact* dyadic rationals
        without torch and without any rounding at the file boundary.
        """
        with open(path) as f:
            blob = json.load(f)
        sd = {name: _hex_to_float(v) for name, v in blob["state_dict_hex"].items()}
        return cls(blob["cfg"], sd)


def _hex_to_float(x):
    if isinstance(x, str):
        return float.fromhex(x)
    return [_hex_to_float(e) for e in x]


# --------------------------------------------------------------------------- #
# forward pass (interval-valued)
# --------------------------------------------------------------------------- #
_ZERO = Fraction(0)


def _linear(W: List[List[Fraction]], b: Optional[List[Fraction]],
            x: List[Ival]) -> List[Ival]:
    """y = W x + b  for an exact matrix W (out x in), bias b, interval vector x."""
    out = []
    for o, row in enumerate(W):
        acc = Ival.point(b[o]) if b is not None else Ival(0, 0)
        for j, w in enumerate(row):
            if w != 0:
                acc = acc + x[j].mul_scalar(w)
        out.append(acc)
    return out


def _attention(blk: dict, z: List[List[Ival]], n: int, d: int,
               heads: int, dh: int, scale: Fraction) -> List[List[Ival]]:
    """Causal multi-head attention; returns Wo(ctx) per position."""
    q = [_linear(blk["Wq"], None, z[i]) for i in range(n)]
    k = [_linear(blk["Wk"], None, z[i]) for i in range(n)]
    v = [_linear(blk["Wv"], None, z[i]) for i in range(n)]

    ctx = [[Ival(0, 0) for _ in range(d)] for _ in range(n)]
    for h in range(heads):
        s = h * dh
        for i in range(n):                      # query position
            # attention logits over allowed keys j <= i (causal), scaled by 1/4
            E = []
            for j in range(i + 1):
                dot = Ival(0, 0)
                for c in range(dh):
                    dot = dot + q[i][s + c] * k[j][s + c]
                E.append(exp_ival(dot.mul_scalar(scale)))
            Elo = [e.lo_i for e in E]
            Ehi = [e.hi_i for e in E]
            tot_lo = sum(Elo)
            tot_hi = sum(Ehi)
            for j in range(i + 1):
                # w_j in [Elo/(Elo + sum_{k!=j} Ehi), Ehi/(Ehi + sum_{k!=j} Elo)]
                w_lo = (Elo[j] * _SCALE) // (tot_hi - Ehi[j] + Elo[j])
                w_hi = _ceil_div(Ehi[j] * _SCALE, tot_lo - Elo[j] + Ehi[j])
                wj = Ival(w_lo, w_hi)
                for c in range(dh):
                    ctx[i][s + c] = ctx[i][s + c] + wj * v[j][s + c]
    return [_linear(blk["Wo"], None, ctx[i]) for i in range(n)]


def forward_logits(W: Weights, tokens: Sequence[int]) -> List[Ival]:
    """Interval enclosure of the model's 2 output logits for one input."""
    n, d, heads, dh = W.n, W.d, W.heads, W.dh
    # embeddings: tok(x) + pos  (exact -> point intervals)
    z = [[Ival.point(W.tok[tokens[i]][c] + W.pos[i][c]) for c in range(d)]
         for i in range(n)]
    for blk in W.blocks:
        attn = _attention(blk, z, n, d, heads, dh, W.scale)
        z = [[z[i][c] + attn[i][c] for c in range(d)] for i in range(n)]   # residual
        # position-wise MLP: z = z + mlp2(relu(mlp1(z)))
        new_z = []
        for i in range(n):
            hmid = [a.relu() for a in _linear(blk["mlp1_w"], blk["mlp1_b"], z[i])]
            delta = _linear(blk["mlp2_w"], blk["mlp2_b"], hmid)
            new_z.append([z[i][c] + delta[c] for c in range(d)])
        z = new_z
    # sum-pool over positions
    pooled = [Ival(0, 0) for _ in range(d)]
    for i in range(n):
        for c in range(d):
            pooled[c] = pooled[c] + z[i][c]
    hidden = [a.relu() for a in _linear(W.h1_w, W.h1_b, pooled)]
    return _linear(W.h2_w, W.h2_b, hidden)


def gap_interval(W: Weights, tokens: Sequence[int]) -> Ival:
    """Enclosure of logit[1] - logit[0] (valid-minus-invalid)."""
    logits = forward_logits(W, tokens)
    return logits[1] - logits[0]


def margin_lower_bound(W: Weights, tokens: Sequence[int], circuit_valid: bool):
    """Rigorous lower bound on the decision margin for one input.

    If the circuit says valid, the margin is gap = logit1 - logit0 and we want
    its lower endpoint; if invalid, the margin is -gap and we want -(upper gap).
    A positive value certifies model-argmax == circuit on this input.

    Returns:
        (margin_lo: Fraction, gap: Ival)
    """
    g = gap_interval(W, tokens)
    margin_lo = g.lo if circuit_valid else -g.hi
    return margin_lo, g


# --------------------------------------------------------------------------- #
# whole-domain verification (optionally parallel — forwards are independent)
# --------------------------------------------------------------------------- #
_WORKER: dict = {}


def _init_worker(cfg, sd, precision):
    set_precision(precision)
    _WORKER["W"] = Weights(cfg, sd)


def _worker_margin(args):
    idx, tokens, valid = args
    mlo, g = margin_lower_bound(_WORKER["W"], tokens, valid)
    return idx, mlo, g.max_bits()


def verify_domain(cfg, sd, items, precision: int = PRECISION,
                  jobs: Optional[int] = 1) -> dict:
    """Rigorously bound the decision margin for every input in `items`.

    Args:
        cfg: model config dict.
        sd: state dict as nested floats/Fractions (e.g. from a hex export).
        items: list of (tokens, circuit_valid) over the whole domain.
        precision: dyadic precision (bits) for the interval arithmetic.
        jobs: worker processes; None => os cpu count; 1 => single process.

    Returns:
        dict with the minimum margin lower bound (Fraction), the arg-min index,
        whether every margin is strictly positive, the max endpoint bit-size,
        and the per-input margin lower bounds.
    """
    tasks = [(i, t, v) for i, (t, v) in enumerate(items)]
    if jobs is None:
        jobs = mp.cpu_count()
    if jobs <= 1:
        set_precision(precision)
        W = Weights(cfg, sd)
        results = []
        for i, t, v in tasks:
            mlo, g = margin_lower_bound(W, t, v)
            results.append((i, mlo, g.max_bits()))
    else:
        with mp.Pool(jobs, initializer=_init_worker,
                     initargs=(cfg, sd, precision)) as pool:
            results = pool.map(_worker_margin, tasks, chunksize=8)
    results.sort(key=lambda r: r[0])
    margins = [r[1] for r in results]
    max_bits = max(r[2] for r in results)
    argmin = min(range(len(margins)), key=lambda i: margins[i])
    return dict(
        min_margin=margins[argmin],
        argmin_index=argmin,
        all_positive=all(m > 0 for m in margins),
        max_bits=max_bits,
        count=len(margins),
        margins=margins,
    )
