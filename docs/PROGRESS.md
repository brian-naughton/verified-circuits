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
