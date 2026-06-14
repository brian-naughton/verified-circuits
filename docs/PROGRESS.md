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

## 2026-06-14 — Milestone B: Circuit == Spec, kernel-checked ✅

**Claim proved (Lean 4, kernel-checked):** the extracted circuit equals the Dyck-1
spec on **every input of every length**:

```lean
theorem circuit_eq_spec (s : List Tok) : Circuit.valid s = Spec.isValid s
```

**Method — structural induction, not enumeration.** Both sides are the same
conjunction `finalDepth == 0 ∧ <order condition>` over the same running-depth
sequence, so two facts carry the theorem (`proofs/VerifiedCircuits/Equiv.lean`):

- `finalDepth_eq` — the circuit's running accumulator depth equals the spec's
  summed final depth.
- the two order conditions coincide, both equal to "every running prefix depth
  ≥ 0" (`allNonnegFrom`): `Circuit.violationCountFrom_eq_zero_iff` (the circuit's
  `violation_count == 0`) and `Spec.minPrefixFrom_nonneg_iff` (the spec's
  `min_prefix_depth ≥ 0`), each proved by induction with the running depth
  generalised. `omega` discharges the per-step integer arithmetic.

Because the proof is inductive over `List Tok`, it covers the headline n=16 domain
(and every other length) for free — there is no per-n enumeration.

**Faithfulness.** The Lean `Spec`/`Circuit` are line-referenced images of
`vcirc/dyck.py` / `vcirc/circuit.py`. The build cross-checks the translation: the
Lean spec reproduces the Catalan profile `[1,0,1,0,2,0,5,0,14,0,42]` (n = 0..10),
and circuit == spec on every string up to length 12.

**Trust surface.** Only the Lean kernel + `propext`, `Quot.sound`:
`#print axioms circuit_eq_spec` prints exactly `[propext, Quot.sound]` — **no
`sorryAx`, no `Lean.ofReduceBool`** (so no `native_decide`), and **no Mathlib**
(Lean core only; `lake build` completes in seconds). The Mathlib-free choice is
for build speed, reproducibility, and a small auditable artifact — **not**
soundness: Lean's kernel checks Mathlib developments just the same.

**Adversarial review (Codex / GPT-5.5).** Independently checked faithfulness
(line-by-line vs `dyck.py`/`circuit.py`: token encoding, increment-then-test
order, min init at 0, sum-as-final-depth — no off-by-one), non-vacuity (genuine
all-length equality of the exported decision functions; n=16 is a subset), and
rigor (structural induction, no `native_decide`/`sorry`/custom axiom; reproduced
the green build). Verdict: no flaw. One honesty note adopted — the trust story now
states the domain is the model's binary `{(, )}` token alphabet (see
`proofs/README.md` §Scope).

**Verdict:** Milestone B passed. The chain `Spec == Circuit == Model` is now
complete — B gives `Spec == Circuit` (kernel-checked, all lengths); A2 gives
`Circuit == Model` (exhaustive rigorous-rational certificate, n=10). Next: scale
to n=16 for the A2 certificate, then the pre-publish checklist.

## 2026-06-14 — n=16 scale-up: `Spec == Circuit == Model` at the headline scale ✅

**Exact model.** A `d=32` model (causal · 2 layers · sum-pool · no LayerNorm — the
A1 architecture, unchanged) trained to **exact 100% over the full 65,536-input
domain**, single seed, first try, minibatched (`--batch 2048`); min decision
margin **+8.095** — *stronger* than n=10 (+6.40) and n=12, so the feared
margin-tightening at longer length did not occur. Saved `models/dyck16_exact_seed0.pt`.

**`Circuit == Model` certified on every one of the 65,536 inputs.**

| Rung | Result |
|---|---|
| v1 (float32 evidence) | argmax == circuit on all 65,536; confusion diagonal (1430 valid = Catalan(8) / 64,106 invalid); min margin +8.0951 |
| v2 (rigorous interval proof) | min-margin lower bound **+8.095074** (= 320678931157985547263444506219 / 2^96), all inputs positive; 96-bit precision; endpoint bits ≤ **108** (n=10 was 105 — no blow-up at 6× the length); ~2 h on 6 cores |
| torch-free re-check (`check.py --jobs=6`) | **VERIFIED** — reproduced +8.095074 from the weight export alone, stdlib + `vcirc/exact.py` only; ~1 h 56 m |

Margin (+8.095) vs interval enclosure width (~1e-21): safety factor ~1e20, so
96-bit precision was ample. Milestone B's Lean proof already covers n=16
(length-generic), so **the full `Spec == Circuit == Model` chain now holds at the
headline scale.**

**Pre-publish corroborations (three cleanly-separated claims).** Certificate
soundness is **analytic** (every `vcirc/exact.py` op rounds outward → rigorous
enclosure by construction); the two empirical checks *corroborate*, they are not
the guarantee:

- **Purpose-1 — enclosure corroboration** (`experiments/validate_exact.py`, now
  `--model` n-general): a high-precision float64 eval of the exact-real function
  lies inside the v2 interval within float64's own error. n=10 fidelity 2.8e-14;
  n=16 fidelity 3.3e-7; enclosure 23/23 sampled both; worst sampled margins match
  the certs (+6.395669, +8.095074).
- **Purpose-2 — float32 faithfulness** (`experiments/faithfulness.py`): (a)
  decision agreement — float32 argmax == circuit on **every** input (full domain,
  both n); (b) numerical fidelity — `|float32 gap − exact gap|` = 3.3e-5 (n=10),
  2.7e-4 (n=16), within the 1e-3 float-error scale. (The enclosure width ~1e-21 is
  far tighter than float32's ~1e-4 rounding, so a *literal* float32 ∈ [lo,hi]
  containment is neither expected nor claimed — faithfulness = agreement + fidelity.)

**Reviewer FAQ** (`docs/FAQ.md`): the three-link trust table; the three
corroboration claims above; how to self-verify; the ownable gap vs Gross et al. /
Hadad et al.; and honest limitations (tiny model / finite task; the transcription
link is the one corroborated-not-proved seam; novelty = mechanism + exact
checkability, not the accuracy number).

**Verdict:** n=16 scale-up + pre-publish checklist done. Remaining before going
public: the explainer / in-repo "preprint" (Leg 3 of `docs/N16-PLAN.md`, written
last once all numbers are final), then set up the public remote.

Reproduce: `python -m vcirc.certify --rung v2 --jobs 6 --model models/dyck16_exact_seed0.pt`
then `python certificates/check.py certificates/dyck16_exact_seed0.v2.cert.json --jobs=6`.

Reproduce: `cd proofs && lake build`.
