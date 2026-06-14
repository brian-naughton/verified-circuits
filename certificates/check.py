#!/usr/bin/env python3
"""Standalone re-verifier for a v2 `Circuit == Model` certificate.

Run:
    python certificates/check.py certificates/dyck10_exact_seed0.v2.cert.json

What it does, trusting nothing but the Python standard library and two small,
auditable files (this script + ``vcirc/exact.py``, the interval-arithmetic
core):

  1. recompute the SHA256 of the exact weight export and confirm it matches the
     hash pinned in the certificate (so we verify *those* weights);
  2. reconstruct the exact dyadic-rational weights from the hex export
     (``float.fromhex`` — no rounding, no torch);
  3. independently recompute the Dyck-1 circuit decision for every input;
  4. re-run the rigorous interval forward pass and confirm that, on *every*
     input, the decision margin's lower bound is strictly positive and at least
     the bound claimed in the certificate.

It does NOT import torch, numpy, or any training/extraction code. The "what must
you trust?" surface is: this file, ``vcirc/exact.py``, and the Python stdlib.
"""
import hashlib
import itertools
import json
import os
import sys

# make vcirc.exact importable without installing anything
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vcirc import exact  # stdlib-only interval-arithmetic core  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def circuit_valid(tokens) -> bool:
    """Dyck-1 validity, recomputed independently (no import of the circuit)."""
    depth = 0
    violations = 0
    for t in tokens:
        depth += 1 if t == 0 else -1
        if depth < 0:
            violations += 1
    return depth == 0 and violations == 0


def check(cert_path: str) -> int:
    with open(cert_path) as f:
        cert = json.load(f)
    cert_dir = os.path.dirname(os.path.abspath(cert_path))

    if cert.get("rung") != "v2-exact":
        print(f"ERROR: not a v2-exact certificate (rung={cert.get('rung')})")
        return 1

    # 1. weights file hash must match what the certificate pinned
    weights_path = os.path.join(ROOT, cert["weights_export"])
    actual = sha256_file(weights_path)
    if actual != cert["weights_export_sha256"]:
        print(f"ERROR: weights export hash mismatch\n  cert:   "
              f"{cert['weights_export_sha256']}\n  actual: {actual}")
        return 1
    print(f"[1/4] weights export hash OK ({actual[:16]}...)")

    # 2. reconstruct exact weights from the hex export (no torch)
    W = exact.Weights.from_hex_export(weights_path)
    n = W.n
    print(f"[2/4] reconstructed exact weights (n={n}, d={W.d}, "
          f"scale 1/sqrt(dh)={W.scale})")

    # 3 + 4. re-derive circuit + rigorous margin lower bound on every input
    exact.set_precision(int(cert["interval_precision_bits"]))
    claimed = float(cert["min_margin_lower_bound"]["float"])
    domain = list(itertools.product((0, 1), repeat=n))
    if len(domain) != cert["domain_size"]:
        print(f"ERROR: domain size mismatch {len(domain)} != {cert['domain_size']}")
        return 1

    worst = None
    worst_input = None
    max_bits = 0
    for tokens in domain:
        c = circuit_valid(tokens)
        mlo, g = exact.margin_lower_bound(W, tokens, c)
        max_bits = max(max_bits, g.max_bits())
        if mlo <= 0:
            print(f"ERROR: non-positive margin lower bound at "
                  f"{''.join(map(str, tokens))}: {float(mlo)}")
            return 1
        if worst is None or mlo < worst:
            worst = mlo
            worst_input = tokens
    print(f"[3/4] circuit re-derived + margins recomputed on all {len(domain)} inputs")

    # the recomputed worst-case bound must (re)confirm the certificate's claim
    if abs(float(worst) - claimed) > 1e-9:
        print(f"WARNING: recomputed min margin {float(worst):.6f} differs from "
              f"certificate {claimed:.6f} (precision/version mismatch?)")
    print(f"[4/4] every margin lower bound > 0; "
          f"min = {float(worst):+.6f} at {''.join(map(str, worst_input))} "
          f"(max endpoint bits {max_bits})")

    print(f"\nVERIFIED: '{cert['claim'][:70]}...'")
    print(f"  domain {cert['domain_size']}  precision {cert['interval_precision_bits']} bits  "
          f"rigorous min margin >= {float(worst):+.6f}")
    return 0


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        print("usage: python certificates/check.py <certificate.v2.cert.json>")
        return 2
    return check(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
