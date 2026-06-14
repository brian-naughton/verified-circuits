#!/usr/bin/env python3
"""Probe 2 — Violation encoding.

Does the layer-2 output (z2) linearly encode a per-position flag for
"prefix went negative" (d_i < 0)? Also test the running-minimum hypothesis
and compare against z1 (after block0) to localise where the signal is created.

Read-only analysis over the full length-10 Dyck-1 domain (1024 strings).
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression

CACHE = "experiments/cache/dyck10_exact_seed0_acts.npz"
SEED = 0


def flatten_positions(z: np.ndarray) -> np.ndarray:
    """(N,10,32) -> (N*10,32) feature matrix over all (input,position) pairs."""
    n, t, d = z.shape
    return z.reshape(n * t, d)


def eval_classifier(name: str, clf, X: np.ndarray, y: np.ndarray) -> dict:
    """Fit a binary classifier and report imbalance-aware metrics."""
    clf.fit(X, y)
    pred = clf.predict(X)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    acc = float(np.mean(pred == y))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"  [{name}] acc={acc:.6f}  pos-precision={prec:.4f}  pos-recall={rec:.4f}"
          f"  (tp={tp} fp={fp} fn={fn})")
    return {"acc": acc, "prec": prec, "rec": rec, "clf": clf}


def normalised_margin(svc: LinearSVC, X: np.ndarray, y: np.ndarray) -> float:
    """Geometric margin of the fitted LinearSVC on the (standardised) features.

    margin = min_i y_i' * (w.x_i + b) / ||w||, with y' in {-1,+1}.
    """
    w = svc.coef_.ravel()
    b = float(svc.intercept_.ravel()[0])
    f = X @ w + b
    yy = np.where(y == 1, 1.0, -1.0)
    func_margin = yy * f
    return float(func_margin.min() / np.linalg.norm(w))


def main() -> None:
    rng = np.random.default_rng(SEED)
    d = np.load(CACHE)
    z1, z2 = d["z1"], d["z2"]            # (1024,10,32)
    depth, viol = d["depth"], d["viol"]  # (1024,10) int

    n, t, _ = z2.shape
    total = n * t

    # ---- 1. Class balance -------------------------------------------------
    y = viol.reshape(total).astype(int)
    pos = int(y.sum())
    print("=" * 70)
    print(f"1. CLASS BALANCE")
    print(f"   total (input,position) pairs = {total}")
    print(f"   viol==1 count = {pos}   base rate = {pos/total:.6f}")
    print(f"   inputs with any violation = {int((viol.sum(1) > 0).sum())}/{n}")

    # ---- 2. Linear separability of d_i<0 flag on z2 -----------------------
    print("=" * 70)
    print("2. VIOLATION-FLAG LINEAR SEPARABILITY (z2)")
    X2 = flatten_positions(z2)
    # standardise for a meaningful, scale-free margin
    mu, sd = X2.mean(0), X2.std(0) + 1e-12
    X2s = (X2 - mu) / sd

    svc2 = LinearSVC(C=1.0, max_iter=200000, dual=True)
    r_svc2 = eval_classifier("z2 LinearSVC", svc2, X2s, y)
    sep2 = abs(r_svc2["acc"] - 1.0) < 1e-12
    margin2 = normalised_margin(svc2, X2s, y) if sep2 else float("nan")
    print(f"   linearly separable (SVC train acc==1)?  {sep2}"
          + (f"   normalised margin = {margin2:.4f}" if sep2 else ""))

    lr2 = LogisticRegression(C=1.0, max_iter=200000, class_weight=None)
    eval_classifier("z2 LogReg", lr2, X2s, y)
    lr2b = LogisticRegression(C=1.0, max_iter=200000, class_weight="balanced")
    eval_classifier("z2 LogReg(balanced)", lr2b, X2s, y)

    # ---- 3. Running-minimum probe (z2 -> m_i) -----------------------------
    print("=" * 70)
    print("3. RUNNING-MIN PROBE (z2 -> m_i = min(d_1..d_i))")
    m = np.minimum.accumulate(depth, axis=1).reshape(total).astype(float)
    reg = LinearRegression()
    reg.fit(X2, m)              # raw features; affine map, scale irrelevant
    pred_m = reg.predict(X2)
    ss_res = float(np.sum((m - pred_m) ** 2))
    ss_tot = float(np.sum((m - m.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    exact = float(np.mean(np.round(pred_m) == m))
    print(f"   running-min range: [{int(m.min())}, {int(m.max())}]")
    print(f"   R^2 = {r2:.6f}   exact-recoverability round(pred)==m_i = {exact:.6f}")

    # Does sign(running-min) recover the violation flag? (m_i<0 == viol_i)
    viol_from_m = (m < 0).astype(int)
    agree = float(np.mean(viol_from_m == y))
    print(f"   (sanity) [m_i<0]==viol_i agreement = {agree:.6f}")

    # ---- 4. Baseline z1 (after block0) ------------------------------------
    print("=" * 70)
    print("4. VIOLATION-FLAG ON z1 (after block0) — is signal created in block1?")
    X1 = flatten_positions(z1)
    mu1, sd1 = X1.mean(0), X1.std(0) + 1e-12
    X1s = (X1 - mu1) / sd1

    svc1 = LinearSVC(C=1.0, max_iter=200000, dual=True)
    r_svc1 = eval_classifier("z1 LinearSVC", svc1, X1s, y)
    sep1 = abs(r_svc1["acc"] - 1.0) < 1e-12
    margin1 = normalised_margin(svc1, X1s, y) if sep1 else float("nan")
    print(f"   linearly separable on z1?  {sep1}"
          + (f"   normalised margin = {margin1:.4f}" if sep1 else ""))
    lr1b = LogisticRegression(C=1.0, max_iter=200000, class_weight="balanced")
    r_lr1b = eval_classifier("z1 LogReg(balanced)", lr1b, X1s, y)

    # running-min on z1 too
    reg1 = LinearRegression(); reg1.fit(X1, m)
    pm1 = reg1.predict(X1)
    r2_1 = 1.0 - np.sum((m - pm1) ** 2) / ss_tot
    exact1 = float(np.mean(np.round(pm1) == m))
    print(f"   z1 running-min: R^2={r2_1:.6f}  exact-recoverability={exact1:.6f}")

    # ---- SUMMARY ----------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print(f"  base rate viol==1 .......... {pos/total:.6f} ({pos}/{total})")
    print(f"  z2 SVC: acc={r_svc2['acc']:.5f} prec={r_svc2['prec']:.4f} "
          f"rec={r_svc2['rec']:.4f} separable={sep2} margin={margin2:.4f}")
    print(f"  z2 running-min: R^2={r2:.5f} exact={exact:.5f}")
    print(f"  z1 SVC: acc={r_svc1['acc']:.5f} prec={r_svc1['prec']:.4f} "
          f"rec={r_svc1['rec']:.4f} separable={sep1}")
    print(f"  z1 LogReg(bal) pos-recall={r_lr1b['rec']:.4f} "
          f"vs z2 SVC pos-recall={r_svc2['rec']:.4f}")
    print(f"  z1 running-min: R^2={r2_1:.5f} exact={exact1:.5f}")


if __name__ == "__main__":
    main()
