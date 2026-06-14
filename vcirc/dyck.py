"""Dyck-1 (balanced parentheses) over fixed-length binary strings — the SPEC.

Token encoding: 0 = '(' (depth +1), 1 = ')' (depth -1).
A length-n string is a tuple of n ints in {0,1}.

This is the GROUND-TRUTH SPEC: the trained model's labels come from `is_valid`,
and the eventual Lean theorem will prove an extracted circuit equals this spec
over the entire finite domain. Pure standard library — no torch.
"""

from itertools import product
from typing import Iterator, List, Tuple

Token = int
String = Tuple[int, ...]


def step(tok: Token) -> int:
    """Depth increment for a token: '(' -> +1, ')' -> -1."""
    return 1 if tok == 0 else -1


def depth_profile(s: String) -> List[int]:
    """Running depth after each prefix (length n; excludes the empty prefix)."""
    d, out = 0, []
    for t in s:
        d += step(t)
        out.append(d)
    return out


def final_depth(s: String) -> int:
    return sum(step(t) for t in s)


def min_prefix_depth(s: String) -> int:
    d, m = 0, 0
    for t in s:
        d += step(t)
        if d < m:
            m = d
    return m


def is_valid(s: String) -> bool:
    """True iff s is balanced: depth never goes negative and ends at 0."""
    return final_depth(s) == 0 and min_prefix_depth(s) >= 0


def enumerate_all(n: int) -> Iterator[String]:
    """All 2^n length-n strings."""
    return product((0, 1), repeat=n)


def count_valid(n: int) -> int:
    return sum(1 for s in enumerate_all(n) if is_valid(s))
