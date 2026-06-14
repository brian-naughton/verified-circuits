"""Validate vcirc.exact (the v2 interval core) against a float64 recompute.

Checks, on a sample of the domain:
  * fidelity: the interval midpoint matches a float64 evaluation of the same
    exact weights (so the architecture was transcribed correctly);
  * enclosure: the float64 gap lies inside [lo, hi] (within float64's own error);
  * margin lower bounds are positive and agree in sign with the circuit;
  * interval width, endpoint bit-size, and per-input timing at PRECISION=96.

Run: python -u -m experiments.validate_exact
"""
import argparse
import sys
import time

import torch

from vcirc import dyck, exact
from vcirc.certify import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dyck10_exact_seed0.pt")
    ap.add_argument("--sample", type=int, default=21,
                    help="inputs sampled for the (heavier) exact enclosure check")
    args = ap.parse_args()

    model = load_model(args.model)
    n = model.cfg["n"]
    sd = {k: v.tolist() for k, v in model.state_dict().items()}
    W = exact.Weights(model.cfg, sd)

    strings = list(dyck.enumerate_all(n))
    X = torch.tensor(strings, dtype=torch.long)
    with torch.no_grad():
        L64 = load_model(args.model).double()(X).numpy()
    gap64 = L64[:, 1] - L64[:, 0]

    # sample dynamically (n-general): the tightest-margin input (worst case) plus
    # a spread across the domain — no hard-coded input string.
    circ = [1 if dyck.is_valid(s) else 0 for s in strings]
    margins64 = [gap64[i] if circ[i] else -gap64[i] for i in range(len(strings))]
    tight = min(range(len(strings)), key=lambda i: margins64[i])
    step = max(1, len(strings) // max(1, args.sample))
    sample = sorted(set([tight] + list(range(0, len(strings), step))))

    exact.set_precision(96)
    print(f"model={args.model}  n={n}  PRECISION={exact.PRECISION} bits;  "
          f"scale 1/sqrt(dh) = {W.scale}")

    maxdiff = 0.0
    enclosed = 0
    maxbits = 0
    worst = None
    worst_in = None
    t0 = time.time()
    for i in sample:
        s = strings[i]
        g = exact.gap_interval(W, s)
        mid = float((g.lo + g.hi) / 2)
        maxdiff = max(maxdiff, abs(mid - gap64[i]))
        if float(g.lo) - 1e-9 <= gap64[i] <= float(g.hi) + 1e-9:
            enclosed += 1
        maxbits = max(maxbits, g.max_bits())
        mlo = g.lo if circ[i] else -g.hi
        if worst is None or mlo < worst:
            worst, worst_in = mlo, s
    dt = (time.time() - t0) / len(sample)
    print(f"fidelity:  max|exact_mid - float64_gap| = {maxdiff:.3e}")
    print(f"enclosure: float64 gap in [lo,hi] (+-1e-9): {enclosed}/{len(sample)}")
    print(f"gap endpoint max bits: {maxbits};  per-input ~{dt*1000:.0f}ms  "
          f"-> full domain ({len(strings)}) ~ {dt*len(strings):.1f}s")
    print(f"worst sampled margin lower bound: {float(worst):+.6f} "
          f"at {''.join(map(str, worst_in))}  (must stay > 0 on the full domain)")

    if maxdiff > 1e-6 or enclosed < len(sample) or worst <= 0:
        print("VALIDATION FAILED")
        sys.exit(1)
    print("VALIDATION OK")


if __name__ == "__main__":
    main()
