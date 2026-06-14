"""Train TinyTransformer on Dyck-1 validity; evaluate over the ENTIRE domain.

Run:  python -m vcirc.train --n 10 --seeds 6
Goal: an exact (100% full-domain) model with positive decision margins, saved to
models/, ready for circuit extraction + certification.
"""
import argparse
import os
import random

import torch
import torch.nn as nn

from vcirc import dyck
from vcirc.model import TinyTransformer

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def make_domain(n):
    strings = list(dyck.enumerate_all(n))
    X = torch.tensor(strings, dtype=torch.long)
    y = torch.tensor([int(dyck.is_valid(s)) for s in strings], dtype=torch.long)
    return X, y, strings


def run_one(args, seed):
    torch.manual_seed(seed); random.seed(seed)
    X, y, strings = make_domain(args.n)
    N = len(strings); npos = int(y.sum())
    idx_pos = [i for i in range(N) if y[i] == 1]
    idx_neg = [i for i in range(N) if y[i] == 0]
    random.shuffle(idx_pos); random.shuffle(idx_neg)
    f = args.trainfrac
    tr = idx_pos[: int(f * len(idx_pos))] + idx_neg[: int(f * len(idx_neg))]
    random.shuffle(tr); tr_t = torch.tensor(tr)

    model = TinyTransformer(args.n, args.d, args.heads, args.ff, args.layers,
                            not args.nocausal, args.pool)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    w = torch.tensor([N / (2 * (N - npos)), N / (2 * npos)], dtype=torch.float)
    lossf = nn.CrossEntropyLoss(weight=w)

    for _ in range(args.epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(X[tr_t]), y[tr_t])
        loss.backward(); opt.step(); sched.step()

    model.eval()
    with torch.no_grad():
        logits = model(X); pred = logits.argmax(1)
    wrong = int((pred != y).sum())
    gap = (logits.gather(1, y[:, None]).squeeze(1)
           - logits.gather(1, (1 - y)[:, None]).squeeze(1))
    return wrong, gap.min().item(), N, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--ff", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pool", default="sum")
    ap.add_argument("--nocausal", action="store_true")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--trainfrac", type=float, default=0.85)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    print(f"n={args.n} d={args.d} heads={args.heads} layers={args.layers} "
          f"causal={not args.nocausal} pool={args.pool} trainfrac={args.trainfrac}")
    best = None
    for s in range(args.seeds):
        wrong, margin, N, model = run_one(args, s)
        tag = "  <-- 100%!" if wrong == 0 else ""
        print(f"  seed {s}: {N - wrong}/{N} correct  ({wrong} wrong)  "
              f"min-margin {margin:+.3f}{tag}")
        if best is None or wrong < best[0]:
            best = (wrong, margin, model, s)
    w, m, model, s = best
    print(f"\nBEST: seed {s}  {w} wrong  min-margin {m:+.3f}")
    if w == 0:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, f"dyck{args.n}_exact_seed{s}.pt")
        torch.save({"state_dict": model.state_dict(), "cfg": model.cfg}, path)
        print(f"saved exact model -> {path}  (exact 100% full-domain, positive margin)")


if __name__ == "__main__":
    main()
