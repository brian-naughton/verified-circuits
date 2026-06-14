# Progress

## 2026-06-14 — Foothold-A1: exact model ✅

**Question:** can a tiny, transparent transformer learn Dyck-1 validity to
*exact 100%* over the entire finite domain (with positive decision margins), so
there is something exactly-certifiable to extract a circuit from?

**First attempt (mean-pool, bidirectional, 1 layer):** plateaued at ~99.8–99.9%.
Across an 8-seed sweep the best had 1 error; *every* error was the order /
min-prefix check (`valid-acc` always 100%; some invalid strings classified
valid). The architecture could count depth but not *exactly* test "did a prefix
go negative?".

**Redesign (causal · 2 layers · sum-pool · MLP head):** principled inductive bias —
causal attention computes per-position prefix depth, the second layer detects
per-position violations, sum-pool aggregates final depth + violation count, the
MLP head forms the `|final depth| == 0` test.

**Result:**

| n | domain | seeds at exact 100% | min margin range |
|---|---|---|---|
| 10 | 1,024 | 6 / 6 | +5.7 … +7.9 |
| 12 | 4,096 | 3 / 3 | +4.5 … +7.5 |

Robust (every seed), not fragile; healthy margins (decision-level certification
has ample room). The principled redesign working first try is itself a signal
that the model implements the depth→violation→aggregate decomposition — i.e. an
*extractable* circuit. Exact checkpoints saved under `models/`.

**Verdict:** A1 passed. Next: A2 — extract the symbolic circuit and certify
`Circuit == Model` argmax on every input.

Reproduce: `python -m vcirc.train --n 10 --seeds 6`.

## 2026-06-14 — A2a: circuit extraction ✅

**Question:** does the exact model (`dyck10_exact_seed0.pt`, full-domain exact,
min margin +6.40) *internally* implement the hypothesised circuit
`valid = (no prefix depth < 0) AND (final depth == 0)` — so the symbolic program
is a faithful **circuit**, not just a black-box reimplementation of the spec?

**Method.** One activation cache over all 1,024 inputs
(`experiments/extract_activations.py`) feeding three bounded linear probes
(`experiments/probe{1,2,3}_*.py`). All numbers below are over the full domain.

| Probe | Finding | Numbers |
|---|---|---|
| **Depth** (per-position residual → `d_i`) | running prefix depth is (near-)exactly linearly decodable **per position** | per-position R² **0.999–1.000**; shared-direction probe only 0.876 (encoding direction is position-dependent) |
| **Violation** (residual → `[d_i<0]`) | the per-position "prefix went negative" flag is **perfectly linearly separable** | LinearSVC **100%**, precision/recall 1.0; the crisp signal is the instantaneous flag, not running-min (R²≈0.87) |
| **Aggregation/readout** (`pooled`, logits) | sum-pool exposes the two aggregates; readout fires valid iff both are zero | `pooled→final_depth` R² **0.99999** (exact); `pooled→violation_count` R² **0.991**; predicted-valid **1.0 only** in the (final=0, viol=0) cell, gap **+10.4**; logit-gap penalises both `|final_depth|` and `violation_count` |

**Honest nuance:** both depth and the violation flag are already established after
**block 0** (block 1 does not sharpen them) — causal attention can prefix-sum and
the block-0 MLP can threshold within a single block. The hypothesis pinned depth
to "layer 1" and violations to "layer 2"; reality is that block 0 already forms
both per-position features and the pool+readout does the rest. The mechanism
(depth → per-position violation flag → sum-pool aggregates → non-monotone
`==0` readout) holds; the layer attribution was looser than hypothesised.

**Extracted circuit:** `vcirc/circuit.py` — `Circuit.trace`/`Circuit.eval` expose
exactly those intermediates (per-position depths, violation flags, `final_depth`,
`violation_count`, decision). It equals the spec by construction over the full
domain for n ≤ 14 (Catalan-verified). The substantive A2 claim — that the trained
**model** equals this circuit on every input, in exact arithmetic — is A2b.

**Verdict:** A2a passed. Next: A2b — exhaustive, exact-arithmetic
`Circuit == Model` certificate.

## 2026-06-14 — A2b: Circuit == Model certificate ✅

**Claim certified:** for every input `x ∈ {0,1}^10`, the model's argmax decision
equals `Circuit.eval(x)` with a strictly positive decision margin.

Two rungs, both passing on `dyck10_exact_seed0` (1,024 inputs):

- **v1 (evidence):** the deployed float32 model, run over the full domain.
  Agreement on all 1,024 inputs; confusion perfectly diagonal (42 valid / 982
  invalid); empirical min margin **+6.3957** at `0001011110`.
  `vcirc/certify.py` refuses to emit on any disagreement (the moment of truth).
- **v2 (proof — the differentiator):** the *exact-real function* defined by the
  weights, evaluated in **rigorous interval arithmetic**. A rational lower bound
  on the margin is **strictly positive on every input**: min **≥ +6.3957** (=
  `506717081843543320754521061633 / 2^96`). `argmax == circuit` on all inputs.

**Why v2 is tractable / honest about method.** Weights are exact dyadic rationals
(float32), inputs are integers, and the attention scale `1/√dh = 1/4` is exact —
so the *only* transcendental in the decision path is `exp` inside the two
softmaxes, enclosed by a Taylor+remainder bound. After the first softmax the
computation is interval-valued (rational endpoints). To stop numerator/
denominator bit-sizes ballooning through the two layers, endpoints are rounded
**outward** to a fixed dyadic precision (96 bits) — a *verified enclosure*, not
exact rationals, but still a rigorous lower bound. Endpoint size stayed bounded
at **105 bits** (no blow-up); a fully-exact `Fraction` forward was confirmed to
explode in bit-size, which is why fixed-precision interval arithmetic is used.

**Engineering note.** The interval core is fixed-point integer arithmetic in
units of `2^-96` (`vcirc/exact.py`), ~180 ms/input single-threaded (~13× faster
than naive `Fraction`); generation runs in parallel.

**Trust surface.** Weights are exported as exact `float.hex()`; a standalone
`certificates/check.py` re-verifies the rational margin lower bound from the
weight export alone, depending only on the Python stdlib + the ~390-line
`vcirc/exact.py` core — no torch, no training/extraction code. The deployed
float32 execution is corroborated exhaustively by v1.

**Framing (carried forward).** The certificate proves the *exact-real function*
defined by the weights is correct (margin > 0) on every input [portable]; the
float32 implementation reproduces those decisions exhaustively [v1]. We do not
claim "we proved a neural network is correct" unqualified. The novelty is the
mechanism + exact checkability, not the 100% accuracy.

**Verdict:** A2b passed — A2 complete. Next: Milestone B (Lean 4 proof
`∀x, Circuit.eval x = Spec.eval x` by induction), then scale to n=16.
