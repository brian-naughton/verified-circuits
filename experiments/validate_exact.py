"""Validate vcirc.exact (the v2 interval core) against a float64 recompute.

Checks, on a sample of the domain:
  * fidelity: the interval midpoint matches a float64 evaluation of the same
    exact weights (so the architecture was transcribed correctly);
  * enclosure: the float64 gap lies inside [lo, hi] (within float64's own error);
  * margin lower bounds are positive and agree in sign with the circuit;
  * interval width, endpoint bit-size, and per-input timing at PRECISION=96.

Run: python -u -m experiments.validate_exact
"""
import sys
import time

import torch

from vcirc import dyck, exact
from vcirc.certify import load_model


def main():
    model = load_model("models/dyck10_exact_seed0.pt")
    n = model.cfg["n"]
    sd = {k: v.tolist() for k, v in model.state_dict().items()}
    W = exact.Weights(model.cfg, sd)

    strings = list(dyck.enumerate_all(n))
    X = torch.tensor(strings, dtype=torch.long)
    with torch.no_grad():
        L64 = load_model("models/dyck10_exact_seed0.pt").double()(X).numpy()
    gap64 = L64[:, 1] - L64[:, 0]

    mm = tuple(int(c) for c in "0001011110")          # the v1 min-margin input
    valid_idx = [i for i, s in enumerate(strings) if dyck.is_valid(s)][:10]
    inval_idx = [i for i, s in enumerate(strings) if not dyck.is_valid(s)][:10]
    sample = [strings.index(mm)] + valid_idx + inval_idx

    exact.set_precision(96)
    print(f"PRECISION = {exact.PRECISION} bits;  scale 1/sqrt(dh) = {W.scale}")

    maxdiff = 0.0
    enclosed = 0
    maxbits = 0
    t0 = time.time()
    for i in sample:
        s = strings[i]
        g = exact.gap_interval(W, s)
        mid = float((g.lo + g.hi) / 2)
        maxdiff = max(maxdiff, abs(mid - gap64[i]))
        if float(g.lo) - 1e-9 <= gap64[i] <= float(g.hi) + 1e-9:
            enclosed += 1
        maxbits = max(maxbits, g.max_bits())
    dt = (time.time() - t0) / len(sample)
    print(f"fidelity:  max|exact_mid - float64_gap| = {maxdiff:.3e}")
    print(f"enclosure: float64 gap in [lo,hi] (+-1e-9): {enclosed}/{len(sample)}")
    print(f"gap endpoint max bits: {maxbits};  per-input ~{dt*1000:.0f}ms  "
          f"-> full domain (1024) ~ {dt*1024:.1f}s")

    print("\nmargin lower bounds (rigorous):")
    worst = None
    for i in [strings.index(mm)] + valid_idx[:2] + inval_idx[:2]:
        s = strings[i]
        c = dyck.is_valid(s)
        mlo, g = exact.margin_lower_bound(W, s, c)
        print(f"  {''.join(map(str, s))}  valid={int(c)}  "
              f"margin_lo={float(mlo):+.6f}  width={float(g.width()):.2e}")
        if worst is None or mlo < worst:
            worst = mlo
    print(f"\nworst sampled margin lower bound: {float(worst):+.6f}  "
          f"(must stay > 0 on the full domain for the certificate)")

    if maxdiff > 1e-6 or enclosed < len(sample) or worst <= 0:
        print("VALIDATION FAILED")
        sys.exit(1)
    print("VALIDATION OK")


if __name__ == "__main__":
    main()
