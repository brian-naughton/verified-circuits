"""float32-implementation faithfulness check (FAQ corroboration claim #2).

Purpose: corroborate that the *deployed float32 model* reproduces the decisions of
the exact-real function the v2 certificate reasons about. This is distinct from
`validate_exact.py` (claim #1: that the interval brackets the exact-real value).
Neither is the guarantee — the guarantee is analytic (outward rounding in
`vcirc/exact.py`); these corroborate it.

Two things, per docs/FAQ.md:
  (a) decision agreement, FULL domain: float32 argmax == circuit == sign of the
      exact-real margin. (float32==circuit is the v1 certificate; the margin sign
      ==circuit is the v2 certificate; this re-asserts (a) directly and cheaply.)
  (b) numerical fidelity, SAMPLE: |float32 gap - exact-real gap| within the
      float-error scale (the exact-real gap taken as the v2 interval midpoint,
      whose width ~1e-21 is negligible next to float32's ~1e-5 rounding).

Run: python -m experiments.faithfulness --model models/dyck16_exact_seed0.pt --sample 400
"""
import argparse
import sys

import torch

from vcirc import dyck, exact
from vcirc.circuit import Circuit
from vcirc.certify import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dyck10_exact_seed0.pt")
    ap.add_argument("--sample", type=int, default=400,
                    help="inputs for the (heavier) exact fidelity check")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |float32 gap - exact gap| allowed (float-error scale)")
    ap.add_argument("--precision", type=int, default=96)
    args = ap.parse_args()

    model = load_model(args.model)
    n = model.cfg["n"]
    strings = list(dyck.enumerate_all(n))

    # (a) decision agreement over the FULL domain (cheap: float32 + circuit only)
    X = torch.tensor(strings, dtype=torch.long)
    with torch.no_grad():
        logits = model(X)
    f32_arg = logits.argmax(1).tolist()
    f32_gap = (logits[:, 1] - logits[:, 0]).tolist()
    circ = [int(Circuit.eval(s)) for s in strings]
    disagree = [i for i in range(len(strings)) if f32_arg[i] != circ[i]]
    # float32 argmax == circuit  <=>  sign(f32_gap) matches circuit, so this also
    # establishes float32 decision == circuit == (exact margin sign, from the v2 cert)
    print(f"n={n}  domain={len(strings)}")
    print(f"(a) decision agreement float32 argmax == circuit on every input: "
          f"{len(disagree) == 0}  (disagreements: {len(disagree)})")

    # (b) numerical fidelity on a sample (heavier: exact interval forward)
    exact.set_precision(args.precision)
    sd = {k: v.tolist() for k, v in model.state_dict().items()}
    W = exact.Weights(model.cfg, sd)
    # always include the tightest-margin input; then a stride-spread sample
    margins = [g if circ[i] == 1 else -g for i, g in enumerate(f32_gap)]
    tight = min(range(len(strings)), key=lambda i: margins[i])
    step = max(1, len(strings) // max(1, args.sample))
    sample = sorted(set([tight] + list(range(0, len(strings), step))))
    maxdiff = 0.0
    worst_in = None
    for i in sample:
        g = exact.gap_interval(W, strings[i])
        mid = float((g.lo + g.hi) / 2)
        d = abs(mid - f32_gap[i])
        if d > maxdiff:
            maxdiff, worst_in = d, strings[i]
    ok_b = maxdiff <= args.tol
    print(f"(b) fidelity on {len(sample)} inputs: max|float32 gap - exact mid| = "
          f"{maxdiff:.3e}  (tol {args.tol:.0e})  -> {ok_b}")

    if disagree or not ok_b:
        print("FAITHFULNESS CHECK FAILED")
        sys.exit(1)
    print("FAITHFULNESS OK (float32 reproduces the exact-real decisions; "
          "fidelity within float error)")


if __name__ == "__main__":
    main()
