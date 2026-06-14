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
| **A2** — extract a symbolic circuit; certify `Circuit == Model` argmax on every input | ✅ done (mechanism probed; rigorous rational margin ≥ +6.3957 on all 1,024 inputs) |
| **B** — Lean 4 proof `∀ s, Circuit.valid s = Spec.isValid s` (by induction) | ✅ done (kernel-checked, Lean core only; axioms `[propext, Quot.sound]`) |
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
vcirc/          # the package:
                #   dyck.py (spec) · model.py (TinyTransformer) · train.py
                #   circuit.py (extracted symbolic circuit)
                #   certify.py (Circuit == Model: v1 evidence + v2 proof)
                #   exact.py (torch-free rigorous interval-arithmetic core)
                #   export_weights.py (exact float.hex() weight export)
models/         # exact trained checkpoints (tiny; committed for reproducibility)
tests/          # pytest: spec vs Catalan; model exact; circuit == model certificate
experiments/    # activation cache + probes (depth/violation/aggregation)
docs/           # design.md (plan + the ownable gap), PROGRESS.md (results)
proofs/         # (B) Lean 4 project: Circuit == Spec       [scaffold]
certificates/   # (A2) emitted certificates + check.py (standalone re-verifier)
```

## Reproduce

```bash
pip install -e .                 # torch (CPU) + this package
python -m vcirc.train --n 10 --seeds 6     # train; saves an exact model to models/
python -m pytest                 # spec correctness + saved model is exact

# A2 — certify Circuit == Model over the whole domain
python -m vcirc.certify --rung v1                  # exhaustive float32 evidence
python -m vcirc.certify --rung v2 --jobs 4         # rigorous rational margin bound
python certificates/check.py \
    certificates/dyck10_exact_seed0.v2.cert.json   # torch-free re-verification
```

## What must you trust?

We certify the **deployed model directly**: float32 weights are exact dyadic
rationals and the inputs are integers, so there is no "rounded model" to bridge
to. The guarantee has two independently-checkable halves.

- **`Circuit == Model`** (Milestone A2, shipped). The *exact-real function*
  defined by the weights has `argmax == Circuit.eval(x)` with a strictly positive
  decision margin on **every** input — proven by rigorous interval arithmetic
  (rational endpoints, directed/outward rounding; the only transcendental, `exp`
  in the two attention softmaxes, is enclosed by a Taylor+remainder bound; the
  attention scale `1/√dh = 1/4` is exact). The certificate states a rational
  lower bound on the margin (≥ +6.3957 for `dyck10_exact_seed0`), and a tiny
  standalone `certificates/check.py` re-verifies it **from the weight export
  alone** — trusting only the Python standard library and the ~390-line interval
  core (`vcirc/exact.py`); no torch, no training or extraction code. The deployed
  *float32* run is separately corroborated **exhaustively** over the full domain
  (the v1 certificate), so both the portable exact function and the literal
  float32 execution are covered.
- **`Circuit == Spec`** (Milestone B, shipped). A Lean 4 theorem
  `∀ s, Circuit.valid s = Spec.isValid s` proved by **structural induction** (not
  enumeration), so it holds for every length — the headline n=16 domain and all
  others. `lake build` checks it and `#print axioms` reports only `propext,
  Quot.sound` (no `sorry`, no `native_decide`, no Mathlib — Lean core only), so a
  skeptic trusts only the Lean kernel. Reproduce: `cd proofs && lake build`. The
  Mathlib-free build is a speed/reproducibility/auditability choice, not a
  soundness one (the kernel checks Mathlib just the same).

Stated honestly: we do **not** claim "we proved a neural network is correct"
unqualified. We claim the exact-real function these weights define is provably
correct (margin > 0) on every input [portable, rigorous], and that the float32
implementation reproduces those decisions exhaustively on the full finite domain.
The novelty is the *mechanism + exact checkability*, not the accuracy number.

## License

MIT — see [LICENSE](LICENSE).
