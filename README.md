# verified-circuits

**Kernel-checkable mechanistic circuits for learned transformers.**

Train a tiny transformer on a *complete finite task*, reverse-engineer it into a
human-readable symbolic **circuit**, and ship independently-checkable evidence
that

> **Spec  ==  Circuit  ==  Model**   over the **entire** input domain

— not on sampled prompts, but on *every* input, with a positive decision margin,
backed by a kernel-checked proof of the circuit↔spec half.

## Why this is different

Recent work pairs mechanistic interpretability with formal guarantees —
[Gross, Agrawal et al. (NeurIPS 2024)](https://arxiv.org/abs/2406.11779) prove
compact *performance lower bounds*; [Hadad, Katz, Bassan (2026)](https://arxiv.org/abs/2602.16823)
certify circuit *robustness / minimality*. Neither ships a **kernel-checked
circuit-to-spec *equivalence* for a learned algorithm over the whole finite
domain, as a reproducible artefact a skeptic can verify without trusting our
Python.** That is the gap this repo targets.

## Status

| Milestone | State |
|---|---|
| **A1** — a tiny transparent transformer learns the task to *exact 100%* over the full domain, positive margins | ✅ done (Dyck-1, n=10 & n=12, all seeds) |
| **A2** — extract a symbolic circuit; certify `Circuit == Model` argmax on every input | ⏳ next |
| **B** — Lean 4 proof `∀x, Circuit.eval x = Spec.eval x` (by induction) | ⏳ planned |
| Scale to n=16 (headline domain, 65,536 inputs) | ⏳ planned |

See [`docs/PROGRESS.md`](docs/PROGRESS.md) for results and [`docs/design.md`](docs/design.md)
for the full plan.

## The task

**Dyck-1 / balanced parentheses**, fixed length n. Tokens `(`=0, `)`=1. A string
is *valid* iff its running depth never goes negative and ends at 0 — i.e. it
genuinely requires the *order* check, not just a parenthesis count. The spec is
`vcirc/dyck.py` (verified against the Catalan numbers).

## Repository layout

```
vcirc/          # the package: dyck.py (spec), model.py (TinyTransformer), train.py
models/         # exact trained checkpoints (tiny; committed for reproducibility)
tests/          # pytest: spec vs Catalan; saved model is exact over full domain
experiments/    # runnable experiment scripts + notes
docs/           # design.md (plan + the ownable gap), PROGRESS.md (results)
proofs/         # (B) Lean 4 project: Circuit == Spec       [scaffold]
certificates/   # (A2) emitted Circuit == Model certificates [scaffold]
```

## Reproduce

```bash
pip install -e .                 # torch (CPU) + this package
python -m vcirc.train --n 10 --seeds 6     # train; saves an exact model to models/
python -m pytest                 # spec correctness + saved model is exact
```

## What must you trust?

The design separates trust levels deliberately:
- **Kernel-checked (B):** `Circuit == Spec` — a Lean theorem; trust only the Lean kernel.
- **Exhaustively checked (A2):** `Circuit == Model` (argmax) on all inputs — a
  certificate re-checkable by a tiny standalone script in exact arithmetic.
- **Stated honestly:** we certify the *trained (optionally rounded) model*; the
  float→rounded bridge is a separate, optional step.

## License

MIT — see [LICENSE](LICENSE).
