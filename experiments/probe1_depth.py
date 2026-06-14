#!/usr/bin/env python3
"""Probe 1 — Depth encoding.

Tests whether the residual stream after block 1 (z2) linearly encodes the
running prefix depth d_i at each position of a tiny transformer trained to
recognise Dyck-1 over all length-10 binary strings.

This is a representation/recoverability question over the full domain, so we
fit and evaluate on all 1024 strings (no held-out split by design).
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

CACHE_PATH = "experiments/cache/dyck10_exact_seed0_acts.npz"
N_POS = 10


def shared_probe_r2(z: np.ndarray, depth: np.ndarray) -> tuple[float, LinearRegression]:
    """Fit one shared OLS probe over all (input, position) pairs.

    Args:
        z: Residual-stream activations, shape (N, 10, 32).
        depth: Running prefix depth, shape (N, 10).

    Returns:
        Tuple of (overall R^2, fitted model).
    """
    n, npos, dim = z.shape
    x_flat = z.reshape(n * npos, dim)
    y_flat = depth.reshape(n * npos)
    model = LinearRegression(fit_intercept=True)
    model.fit(x_flat, y_flat)
    pred = model.predict(x_flat)
    return r2_score(y_flat, pred), model


def main() -> None:
    d = np.load(CACHE_PATH)
    z1 = d["z1"]      # after block 0
    z2 = d["z2"]      # after block 1
    depth = d["depth"]

    # --- 1. Shared probe on z2 ---
    r2_z2, model_z2 = shared_probe_r2(z2, depth)

    # --- 2. Exact recoverability (round(pred) == d_i) ---
    n, npos, dim = z2.shape
    x_flat = z2.reshape(n * npos, dim)
    y_flat = depth.reshape(n * npos)
    pred_flat = model_z2.predict(x_flat)
    exact = (np.rint(pred_flat).astype(int) == y_flat)
    exact_frac = exact.mean()

    # per-position exact recoverability (using the shared probe), min across pos
    exact_by_pos = exact.reshape(n, npos).mean(axis=0)
    exact_min_pos = exact_by_pos.min()

    # --- 3. Per-position R^2 (fit + eval per position) ---
    per_pos_r2 = []
    for i in range(npos):
        m = LinearRegression(fit_intercept=True)
        m.fit(z2[:, i, :], depth[:, i])
        per_pos_r2.append(r2_score(depth[:, i], m.predict(z2[:, i, :])))
    per_pos_r2 = np.array(per_pos_r2)

    # --- 4. Baseline: shared probe on z1 (after block 0) ---
    r2_z1, _ = shared_probe_r2(z1, depth)

    # --- 5. Sanity: weight L2 norm + residual std (z2 shared probe) ---
    w_norm = float(np.linalg.norm(model_z2.coef_))
    residual_std = float(np.std(y_flat - pred_flat))

    print("=== Probe 1 — Depth encoding ===")
    print(f"[1] Shared-probe R^2 (z2, after block1): {r2_z2:.6f}")
    print(f"[4] Shared-probe R^2 (z1, after block0): {r2_z1:.6f}")
    print(f"    Sharpening (z2 - z1):                {r2_z2 - r2_z1:+.6f}")
    print(f"[2] Exact recoverability (round==d_i):   {exact_frac:.6f}")
    print(f"    Min exact-recoverability over pos:   {exact_min_pos:.6f}")
    print(f"[3] Per-position R^2 min/max:            {per_pos_r2.min():.6f} / {per_pos_r2.max():.6f}")
    print(f"[5] Fitted z2 weight L2 norm:            {w_norm:.6f}")
    print(f"    Residual std (z2 shared probe):      {residual_std:.6f}")


if __name__ == "__main__":
    main()
