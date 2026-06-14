#!/usr/bin/env python3
"""Probe 3 — Aggregation + readout.

Tests whether the sum-pooled residual stream (`pooled`) exposes the two
aggregate statistics that decide Dyck-1 validity — final_depth (d_n) and
violation_count (#positions with d_i < 0) — and whether the MLP readout head
implements the rule  valid := (final_depth == 0 AND violation_count == 0).

This is a representation/mechanism question over the full finite domain, so we
fit and evaluate on all 1024 length-10 strings (no held-out split by design).
"""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from vcirc.model import TinyTransformer

CACHE_PATH = "experiments/cache/dyck10_exact_seed0_acts.npz"
MODEL_PATH = "models/dyck10_exact_seed0.pt"


def fit_probe(pooled: np.ndarray, target: np.ndarray) -> tuple[float, float, LinearRegression]:
    """Fit an OLS probe pooled -> target over the full domain.

    Args:
        pooled: Sum-pooled readout input, shape (N, 32).
        target: Integer aggregate to recover, shape (N,).

    Returns:
        Tuple of (R^2, exact-recoverability fraction, fitted model).
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(pooled, target)
    pred = model.predict(pooled)
    r2 = r2_score(target, pred)
    exact = float((np.rint(pred).astype(int) == target).mean())
    return r2, exact, model


def main() -> None:
    d = np.load(CACHE_PATH)
    pooled = d["pooled"]                 # (1024, 32) sum over positions of z2
    logits = d["logits"]                 # (1024, 2)
    final_depth = d["final_depth"]       # (1024,)  d_n
    violation_count = d["violation_count"]  # (1024,)  #pos with d_i < 0
    label = d["label"]                   # (1024,)  1 iff valid
    n = pooled.shape[0]

    # --- 1. Linear probes pooled -> aggregates ---
    r2_fd, exact_fd, probe_fd = fit_probe(pooled, final_depth)
    r2_vc, exact_vc, probe_vc = fit_probe(pooled, violation_count)

    # --- 2. Readout truth-table: {fd==0} x {vc==0} ---
    gap = logits[:, 1] - logits[:, 0]
    pred_valid = (np.argmax(logits, axis=1) == 1)
    fd_zero = (final_depth == 0)
    vc_zero = (violation_count == 0)

    cells = {
        "fd==0 & vc==0": (fd_zero & vc_zero),
        "fd==0 & vc>0":  (fd_zero & ~vc_zero),
        "fd!=0 & vc==0": (~fd_zero & vc_zero),
        "fd!=0 & vc>0":  (~fd_zero & ~vc_zero),
    }

    print("=== Probe 3 — Aggregation + readout ===")
    print(f"N inputs: {n}")
    print()
    print("[1] Linear probes on pooled (full-domain fit/eval):")
    print(f"    pooled -> final_depth     R^2={r2_fd:.6f}  exact-recover={exact_fd:.6f}")
    print(f"    pooled -> violation_count R^2={r2_vc:.6f}  exact-recover={exact_vc:.6f}")
    print()

    print("[2] Readout truth-table (2x2 cells):")
    print(f"    {'cell':<16}{'count':>7}{'valid-rate':>12}{'mean-gap':>12}")
    for name, mask in cells.items():
        cnt = int(mask.sum())
        if cnt == 0:
            print(f"    {name:<16}{cnt:>7}{'--':>12}{'--':>12}")
            continue
        vr = float(pred_valid[mask].mean())
        mg = float(gap[mask].mean())
        print(f"    {name:<16}{cnt:>7}{vr:>12.4f}{mg:>12.4f}")
    print()

    # Cross-check: model argmax-valid vs ground-truth label
    acc = float((pred_valid.astype(int) == label).mean())
    print(f"    Model argmax-valid vs ground-truth label accuracy: {acc:.6f}")
    print(f"    True-valid count (fd==0 & vc==0): {int((fd_zero & vc_zero).sum())}")
    print()

    # --- 3. Logit-gap regression on aggregate features ---
    feats = np.column_stack([
        final_depth.astype(float),
        np.abs(final_depth).astype(float),
        violation_count.astype(float),
    ])
    reg = LinearRegression(fit_intercept=True)
    reg.fit(feats, gap)
    gap_pred = reg.predict(feats)
    r2_gap = r2_score(gap, gap_pred)
    b0 = float(reg.intercept_)
    b_fd, b_absfd, b_vc = (float(c) for c in reg.coef_)

    print("[3] Logit-gap regression g = b0 + b1*fd + b2*|fd| + b3*vc:")
    print(f"    intercept (b0)            = {b0:+.4f}")
    print(f"    final_depth (b1)          = {b_fd:+.4f}")
    print(f"    abs(final_depth) (b2)     = {b_absfd:+.4f}")
    print(f"    violation_count (b3)      = {b_vc:+.4f}")
    print(f"    R^2                       = {r2_gap:.6f}")
    print()

    # --- 4. Weight inspection: how h1/h2 combine the probe directions ---
    blob = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    m = TinyTransformer(**blob["cfg"])
    m.load_state_dict(blob["state_dict"])
    m.eval()

    W1 = m.h1.weight.detach().numpy()   # (32, 32)  h1_pre = W1 @ pooled + b1
    W2 = m.h2.weight.detach().numpy()   # (2, 32)   logits = W2 @ h1_post + b2

    # Logit-gap row in h1_post space:
    gap_row = W2[1, :] - W2[0, :]       # (32,)

    # Probe directions in pooled space (unit-normalised):
    dir_fd = probe_fd.coef_ / (np.linalg.norm(probe_fd.coef_) + 1e-12)
    dir_vc = probe_vc.coef_ / (np.linalg.norm(probe_vc.coef_) + 1e-12)

    # How does h1 map the pooled probe directions into hidden space?
    # h1_pre response to a unit step along each pooled direction:
    h1_fd = W1 @ dir_fd                 # (32,) hidden-pre shift per unit fd-direction
    h1_vc = W1 @ dir_vc

    # Cosine of the gap_row (which reads h1_post) against the hidden images.
    # (Qualitative; relu sits between, so this is an alignment heuristic.)
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print("[4] Weight inspection:")
    print(f"    gap_row . h1(final_depth dir) cosine     = {cos(gap_row, h1_fd):+.4f}")
    print(f"    gap_row . h1(violation_count dir) cosine = {cos(gap_row, h1_vc):+.4f}")

    # Empirical: correlate logit-gap with the probe-recovered aggregates directly.
    fd_hat = probe_fd.predict(pooled)
    vc_hat = probe_vc.predict(pooled)
    print(f"    corr(gap, |probe final_depth|)           = {np.corrcoef(gap, np.abs(fd_hat))[0,1]:+.4f}")
    print(f"    corr(gap, probe violation_count)         = {np.corrcoef(gap, vc_hat)[0,1]:+.4f}")


if __name__ == "__main__":
    main()
