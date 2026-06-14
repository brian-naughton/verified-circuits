"""Extract per-stage activations of an exact Dyck model over the FULL domain.

Mission-control harness for the A2a probes: runs every 2^n input through the
saved model, captures the per-position residual stream at each stage plus the
pooled/readout activations, and pairs them with the ground-truth circuit
quantities (per-position depth, per-position violation flag, final depth,
violation count). All three probes consume the *same* cache so their findings
are mutually consistent.

Usage:
    python -m experiments.extract_activations --model models/dyck10_exact_seed0.pt

Writes experiments/cache/<modelstem>_acts.npz and verifies that the captured
logits reproduce the model's argmax + min margin exactly.
"""
import argparse
import os

import numpy as np
import torch

from vcirc import dyck
from vcirc.model import TinyTransformer

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def load_model(path: str) -> TinyTransformer:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyTransformer(**blob["cfg"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def circuit_targets(strings: list, n: int) -> dict:
    """Ground-truth per-position + aggregate quantities the probes target."""
    depth = np.zeros((len(strings), n), dtype=np.int64)       # d_i after prefix i
    viol = np.zeros((len(strings), n), dtype=np.int64)        # 1 iff d_i < 0
    for r, s in enumerate(strings):
        depth[r] = dyck.depth_profile(s)
        viol[r] = (depth[r] < 0).astype(np.int64)
    final_depth = depth[:, -1]
    violation_count = viol.sum(1)
    min_prefix = depth.min(1)
    label = np.array([int(dyck.is_valid(s)) for s in strings], dtype=np.int64)
    return dict(depth=depth, viol=viol, final_depth=final_depth,
                violation_count=violation_count, min_prefix=min_prefix, label=label)


def extract(model: TinyTransformer, X: torch.Tensor) -> dict:
    """Run X through the model, capturing per-stage activations via hooks.

    Stages (per position unless noted):
      z0  embeddings  tok(x) + pos
      z1  residual stream after block 0
      z2  residual stream after block 1  (features that get pooled)
      pooled       sum over positions of z2            (readout input)
      h1_pre       h1(pooled)            (readout hidden pre-activation)
      h1_post      relu(h1_pre)
      logits       h2(h1_post)           (2 logits; argmax is the decision)
    """
    caps = {}
    handles = []
    handles.append(model.blocks[0].register_forward_hook(
        lambda m, i, o: caps.__setitem__("z1", o.detach())))
    handles.append(model.blocks[1].register_forward_hook(
        lambda m, i, o: caps.__setitem__("z2", o.detach())))
    handles.append(model.h1.register_forward_pre_hook(
        lambda m, i: caps.__setitem__("pooled", i[0].detach())))
    handles.append(model.h1.register_forward_hook(
        lambda m, i, o: caps.__setitem__("h1_pre", o.detach())))
    handles.append(model.h2.register_forward_hook(
        lambda m, i, o: caps.__setitem__("logits", o.detach())))
    with torch.no_grad():
        z0 = model.tok(X) + model.pos[None, :, :]
        _ = model(X)
    for h in handles:
        h.remove()

    h1_post = torch.relu(caps["h1_pre"])
    out = dict(
        x=X.numpy().astype(np.int64),
        z0=z0.numpy().astype(np.float64),
        z1=caps["z1"].numpy().astype(np.float64),
        z2=caps["z2"].numpy().astype(np.float64),
        pooled=caps["pooled"].numpy().astype(np.float64),
        h1_pre=caps["h1_pre"].numpy().astype(np.float64),
        h1_post=h1_post.numpy().astype(np.float64),
        logits=caps["logits"].numpy().astype(np.float64),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dyck10_exact_seed0.pt")
    args = ap.parse_args()

    model = load_model(args.model)
    n = model.cfg["n"]
    strings = list(dyck.enumerate_all(n))
    X = torch.tensor(strings, dtype=torch.long)

    acts = extract(model, X)
    tgt = circuit_targets(strings, n)
    cache = {**acts, **tgt}

    # --- verification: captured logits must reproduce argmax + min margin ---
    logits = acts["logits"]
    pred = logits.argmax(1)
    label = tgt["label"]
    exact = bool((pred == label).all())
    gap = logits[np.arange(len(label)), label] - logits[np.arange(len(label)), 1 - label]
    min_margin = float(gap.min())

    os.makedirs(CACHE_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.model))[0]
    out_path = os.path.join(CACHE_DIR, f"{stem}_acts.npz")
    np.savez_compressed(out_path, **cache)

    print(f"model={args.model}  n={n}  domain={len(strings)}")
    print(f"stages: " + ", ".join(f"{k}{acts[k].shape}" for k in
          ["z0", "z1", "z2", "pooled", "h1_pre", "logits"]))
    print(f"exact over full domain: {exact}   min margin: {min_margin:+.4f}")
    print(f"valid count: {int(label.sum())} / {len(label)}  "
          f"(Catalan check: {dyck.count_valid(n)})")
    print(f"wrote -> {out_path}")
    if not exact or min_margin <= 0:
        raise SystemExit("ABORT: cache does not reproduce an exact, positive-margin model")


if __name__ == "__main__":
    main()
