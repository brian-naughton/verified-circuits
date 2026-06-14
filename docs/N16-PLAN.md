# n=16 scale-up + run-to-publish plan

**For this session (continuing).** A1 + A2 (n=10) + Milestone B (Lean
`Circuit == Spec`, all lengths) are done and merged to `main`. This is the run to
**publish-ready**: scale the A2 `Circuit == Model` certificate to the headline
n=16 domain, finish the pre-publish checklist, write the explainer, then go
public. Methodology unchanged: subagents for grunt work, Codex as adversary, work
on branches / keep `main` clean, **check in at each seam**.

Branch: `scale-n16`. Public remote stays held until everything below is done.

## Leg 1 — scale `Circuit == Model` to n=16 (65,536 inputs)

Milestone B's Lean proof already covers n=16 (length-generic), so only the
empirical `Circuit == Model` certificate needs to scale.

**1a — train an exact-100% model at n=16.**
- Domain is 2^16 = 65,536 strings (valid count = Catalan(8) = 1,430).
  `train.py` is currently **full-batch**; at this size add **minibatching** to the
  training loop (eval stays full-domain, minibatched forward). Keep the
  architecture (causal · 2 layers · sum-pool · no LayerNorm) — it's the one A1
  validated; only widen (`--d`, `--ff`) or add epochs if needed.
- Seed-sweep; **require exact 100% full-domain with a strictly positive decision
  margin** before certifying (same bar as n=10/12). Expect headwinds: longer
  sequences → more order-violation patterns → possibly tighter margins; may need
  more seeds / a wider model / more epochs.
- Save `models/dyck16_exact_seed{S}.pt` (committed for reproducibility, like n=10).
- **Seam check-in (1): exact n=16 model trained** (report wrong=0, min-margin).

**1b — certify (v1 then v2) over all 65,536 inputs.**
- **v1 (moment of truth):** `python -m vcirc.certify --rung v1 --model
  models/dyck16_exact_seed{S}.pt` — float32 argmax == circuit on every input,
  diagonal confusion, empirical min margin. If any disagreement → the circuit
  isn't faithful at n=16; stop and investigate.
- **v2 (proof):** `python -m vcirc.certify --rung v2 --jobs <cores-2>
  --model …`. ≈64× the inputs of n=10 and heavier per input (16×16 attention),
  so lean on `--jobs`; this may run hours. **Watch the rigorous min-margin lower
  bound doesn't approach the enclosure width** — if it shrinks toward 0, bump
  `--precision` (96 → 128) and re-run. Endpoint bit-size should stay bounded
  (it was ≤105 bits at 96-bit precision, n=10).
- **Re-verify torch-free:** `python certificates/check.py
  certificates/dyck16_exact_seed{S}.v2.cert.json`.
- Update the README result table (n=16 row: domain 65,536, min-margin bound).
- **Seam check-in (2): n=16 v2 cert green + independently re-checked.**

**Fallback (do not let n=16 block).** The method is already fully demonstrated at
n=10/12 + the length-generic Lean proof. **Time-box n=16.** If it resists — won't
reach exact 100%, or the v2 cert is impractical / margins collapse toward the
enclosure even at higher precision — **fall back: n=10 is already a complete,
publishable story.** n=16 is a headline-number upgrade, not a prerequisite.
Document the outcome honestly and move on.

**Kill/▶ at n=16 specifically:** non-positive margin that precision can't rescue;
v2 endpoint bits blow up unbounded; training can't hit exact 100% with a sane
model. → take the fallback.

## Leg 2 — pre-publish checklist

- **Enclosure-soundness insurance:** assert the v2 interval **⊇** the v1 float32
  logits on every input (cheap; if the float logits ever fell outside the proven
  enclosure, the enclosure would be unsound — this catches that). Do it for n=10
  and (if Leg 1 succeeded) n=16.
- **Reviewer FAQ** (`docs/` or README): the **three-link trust table** (kernel /
  interval-cert / corroborated-transcription — already in the README), "what must
  you trust?", and **"vs Gross et al. / Hadad et al."** (the ownable gap).
- **Seam check-in (3): checklist done.**

## Leg 3 — the explainer / in-repo "preprint"  (WRITE LAST)

The missing legibility layer — the thing that makes the repo catch attention. An
accessible, ideally interactive walkthrough so a broad audience can engage beyond
README/code, then trust the proof behind it:
- the problem + the ownable gap;
- the trained model; the extracted **circuit diagram**;
- the mechanism made **visual** (attention heatmaps = prefix-sum; the
  depth/violation probes);
- the **certificate in action** (torch-free `check.py` run + the rigorous margin);
- the **Lean theorem**; the **three-link trust story**.

Format = whatever maximises accessibility (Jupyter notebook, or rendered
markdown/HTML with figures). **Recorded as a planned waypoint now; written only
once the result is final** (after Legs 1–2), so figures/numbers are the published
ones.
- **Seam check-in (4): explainer drafted → mission-control review.**

## Then — go public

Set up the public remote and publish, only once Legs 1–3 are done and signed off.
Pre-publish gates already banked in project memory: complete `Spec==Circuit==Model`
chain, three-link trust table, reviewer FAQ.

## Subagents / adversary

- Grunt work to subagents: the n=16 seed-sweep training run; the long v2 cert run;
  figure generation for the explainer.
- Codex (GPT-5.5) as adversary at the seams: faithfulness of any new code, and a
  skeptical read of the n=16 cert + the enclosure-soundness assertion.
