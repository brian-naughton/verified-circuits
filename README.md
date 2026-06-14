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
| Scale to n=16 (headline domain, 65,536 inputs) | ✅ done (exact 100%; `Circuit == Model` rigorous margin ≥ +8.0950 on all 65,536) |

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

The end-to-end claim `Spec == Circuit == Model` is **three links at three
different trust levels.** They are not equally strong, and "kernel-checked" applies
to exactly one of them — we name all three so a sharp reviewer doesn't have to ask.

| Link | How | Trust level | Scope |
|---|---|---|---|
| `Circuit == Spec` | Lean 4 theorem, by induction | **kernel-proven** (Lean kernel + `propext`, `Quot.sound`) | **all lengths** |
| `Circuit == Model` | rigorous interval-arithmetic certificate | machine-checked rational bound (stdlib + ~390-line interval core) | per-input, **n=10** |
| Lean/Python defs ↔ same algorithm | faithful transcription | **corroborated, not proven** | binary `{(, )}` alphabet |

- **`Circuit == Spec`** (Milestone B). `∀ s, Circuit.valid s = Spec.isValid s`,
  proved by **structural induction** (not enumeration), so it holds for every
  length — n=16 and all others. `#print axioms` reports only `propext, Quot.sound`
  (no `sorry`, no `native_decide`, no Mathlib — Lean core only), so a skeptic
  trusts only the Lean kernel. Reproduce: `cd proofs && lake build`. (Mathlib-free
  is a speed/reproducibility/auditability choice, *not* a soundness one — the
  kernel checks Mathlib just the same.)
- **`Circuit == Model`** (Milestone A2). We certify the **deployed model
  directly**: float32 weights are exact dyadic rationals and inputs are integers,
  so there is no "rounded model" to bridge to. The *exact-real function* the
  weights define has `argmax == Circuit.eval(x)` with a strictly positive decision
  margin on **every** input — rigorous interval arithmetic (rational endpoints,
  directed/outward rounding; the only transcendental, `exp` in the two attention
  softmaxes, enclosed by a Taylor+remainder bound; scale `1/√dh = 1/4` exact).
  Rational margin lower bound ≥ +6.3957 for `dyck10_exact_seed0`;
  `certificates/check.py` re-verifies it **from the weight export alone** (Python
  stdlib + the ~390-line `vcirc/exact.py` core; no torch, no training/extraction
  code). The literal *float32* run is corroborated **exhaustively** by the v1
  certificate. Certified at **n=10** (margin ≥ +6.3957) and the headline
  **n=16** (full 65,536-input domain, rigorous margin ≥ +8.0950, endpoint bits
  ≤ 108 at 96-bit precision).
- **Lean/Python defs ↔ the same algorithm** (the transcription). The Lean
  `Spec`/`Circuit` are faithful images of `vcirc/dyck.py`/`vcirc/circuit.py` — but
  this link is **corroborated, not formally proven** (Python is not a formal
  object, so it *cannot* be). The corroboration is strong and multi-pronged: (1)
  the Lean spec reproduces the **Catalan numbers** `[1,0,1,0,2,0,5,0,14,0,42]` — it
  is validated against what Dyck-1 *mathematically is*, independent of any
  `dyck.py` bug; (2) circuit == spec on every string up to length 12; (3) an
  independent line-by-line audit (Codex / GPT-5.5) against cited Python line
  numbers; (4) each Lean definition cites the Python lines it mirrors. Scope: the
  model's binary `{(, )}` token alphabet (`proofs/README.md` §Scope).

Stated honestly: we do **not** claim "we proved a neural network is correct"
unqualified. The circuit↔spec equivalence is kernel-proven for all lengths; the
exact-real function these weights define provably makes the circuit's decision
(margin > 0) on every input [n=10, portable, rigorous], with the float32 execution
corroborated exhaustively; and the one unavoidable soft seam — that our Lean and
Python transcribe the same algorithm — is corroborated, not proven. The novelty is
the *mechanism + exact checkability*, not the accuracy number.

## License

MIT — see [LICENSE](LICENSE).
