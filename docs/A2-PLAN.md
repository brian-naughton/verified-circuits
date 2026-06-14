# A2 kickoff plan — circuit extraction + certification

**For the next session.** You are picking up project #3 ("verified-circuits") cold.
Orient from `MEMORY.md` (workspace memory) + `README.md` + `docs/design.md` +
`docs/PROGRESS.md`. Foothold-A1 is done: `models/dyck10_exact_seed0.pt` classifies
all 1,024 length-10 Dyck strings correctly with min margin +6.4. **A2 = extract a
human-readable symbolic circuit and certify `Circuit == Model` on every input.**
Recommended execution: this session as mission control, subagents for the
data-heavy probes; you synthesise.

## Goal

Produce (1) a symbolic `Circuit` (a small, readable program) and (2) a certificate
that the trained model's argmax equals `Circuit` on **all** inputs with a positive
margin. (The Lean `Circuit == Spec` half is Milestone B, later.)

## A2a — extraction (exploratory; subagents do bounded probes, return compact findings)

Hypothesised mechanism (the architecture was designed for it — verify, don't assume):
`valid = (no prefix depth < 0) AND (final depth == 0)`.

Probes (one subagent each; each returns numbers + a verdict, not raw dumps):
1. **Depth probe:** does layer-1 output linearly encode running depth `d_i` at each
   position? Fit a linear probe `residual_after_block1[:, i, :] -> d_i` over all
   inputs; report R² / exact recoverability. Expect near-exact.
2. **Violation probe:** does the layer-2 / MLP output encode a per-position flag for
   `d_i < 0` (and/or the running minimum)? Probe for it; report separability.
3. **Aggregation probe:** confirm sum-pool produces features ≈ (final depth,
   violation count); confirm the readout MLP fires "valid" iff final-depth≈0 and
   violation-count≈0. Inspect `h1`/`h2` weights.

Synthesis (mission control): write `vcirc/circuit.py` — the explicit symbolic
`Circuit.eval(x)` (compute depths; return `final==0 and min_prefix>=0`). This
should mirror `dyck.is_valid` *by construction* — that's expected; the content is
showing the **trained model implements it**, certified next.

## A2b — certification (mechanical; ideal for subagent TDD)

Write `vcirc/certify.py` + `tests/test_certify.py`:
- For every input in the domain, compare model argmax to `Circuit.eval`; require
  agreement + a positive logit margin. Emit a certificate (JSON) with: domain
  size, SHA256 of weights + circuit source + checker, per-class counts, min
  margin, checker version → `certificates/`.
- **Rigour ladder (pick the rung, document honestly):**
  - v1 (evidence): high-precision float eval, report empirical min margin.
  - v2 (proof): rigorous **interval arithmetic** through the network (rational
    bounds; softmax bounded by a verified enclosure) proving the margin interval
    is strictly positive on every input → a genuine certificate. This is the
    differentiator; aim here.
- A tiny standalone checker re-verifies a certificate from the weights file alone.

## Kill criteria (from the foothold — abandon/pivot if hit)

- No small circuit matches the model over the full domain.
- Agreement needs a brittle lookup-table, not an algorithm.
- Quantization/rounding destroys margins; interval bounds blow up.
- The story can't be stated more strongly than "we sampled prompts and it worked."

## After A2

Scale to n=16 (65,536 inputs; minibatch training on CPU). Then Milestone B: Lean 4
+ Mathlib proof `∀x, Circuit.eval x = Spec.eval x` by induction. Then polish the
repo (circuit diagram, adversarial FAQ "what must you trust? / vs Gross et al.")
for publishing.
