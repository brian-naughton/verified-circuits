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


def eval_full(model, X, batch):
    """Model logits over the whole domain, minibatched if ``batch > 0`` (so the
    65,536-input n=16 forward fits in memory). Equivalent to ``model(X)``."""
    model.eval()
    with torch.no_grad():
        if batch and batch > 0:
            logits = torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)], 0)
        else:
            logits = model(X)
    return logits


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

    every = max(1, args.epochs // 10) if args.progress else 0
    for ep in range(args.epochs):
        model.train()
        if args.batch and args.batch > 0:
            # minibatch one epoch over the training set (default off: batch=0)
            perm = tr_t[torch.randperm(len(tr_t))]
            for i in range(0, len(perm), args.batch):
                bidx = perm[i:i + args.batch]
                opt.zero_grad()
                loss = lossf(model(X[bidx]), y[bidx])
                loss.backward(); opt.step()
            sched.step()
        else:
            opt.zero_grad()
            loss = lossf(model(X[tr_t]), y[tr_t])
            loss.backward(); opt.step(); sched.step()
        if every and (ep % every == 0 or ep == args.epochs - 1):
            print(f"    seed {seed} epoch {ep:4d}/{args.epochs}  loss {loss.item():.4f}",
                  flush=True)

    logits = eval_full(model, X, args.batch)
    pred = logits.argmax(1)
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
    ap.add_argument("--batch", type=int, default=0,
                    help="minibatch size for training/eval (0 = full-batch; "
                         "use for large domains, e.g. 4096 at n=16)")
    ap.add_argument("--trainfrac", type=float, default=0.85)
    ap.add_argument("--progress", action="store_true",
                    help="print per-seed training loss ~10x over the run")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing models/ checkpoint "
                         "(cert-pinned models are read-only by default)")
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
        name = f"dyck{args.n}_exact_seed{s}.pt"
        path = os.path.join(MODELS_DIR, name)
        # Cert-pinned models in models/ are effectively read-only (a v2 certificate
        # verifies against their SHA256). Don't let a scratch/sweep run silently
        # clobber one — redirect to models/scratch/ unless --force is given.
        if os.path.exists(path) and not args.force:
            scratch = os.path.join(MODELS_DIR, "scratch")
            os.makedirs(scratch, exist_ok=True)
            path = os.path.join(scratch, name)
            print(f"  ({name} already exists and is treated as read-only; saving to "
                  f"models/scratch/ instead — pass --force to overwrite intentionally)")
        torch.save({"state_dict": model.state_dict(), "cfg": model.cfg}, path)
        print(f"saved exact model -> {path}  (exact 100% full-domain, positive margin)")


if __name__ == "__main__":
    main()
