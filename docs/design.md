# Design — verified-circuits

**Date:** 2026-06-14  ·  **Status:** A1 complete; A2 next.

## Goal

A portfolio-grade, publishable artefact at the intersection of mechanistic
interpretability and formal verification, demonstrating:

> a learned transformer's computation rendered as a human-readable symbolic
> **circuit**, with an end-to-end, independently-checkable guarantee that
> `Spec == Circuit == Model` over the **entire** finite input domain.

This sits under the `maths_problems` workspace; selection rationale (max impact ×
hard checkability; career legibility for AI roles) is in
`../HIGH_VALUE_PICKS.md`.

## The ownable gap

- Gross, Agrawal et al., *Compact Proofs of Model Performance via MI*
  (NeurIPS 2024) — compact formal **performance lower bounds** (Max-of-K).
- Hadad, Katz, Bassan (2026) — verifier-certified circuit **robustness /
  minimality** (vision, continuous domains).
- **Us:** kernel-checked **circuit↔spec equivalence** for a *learned algorithm*,
  total finite-domain `Spec == Circuit == Model`, shipped as a reproducible
  artefact. Phrase to own: *"a kernel-checked mechanistic circuit for a learned
  transformer on a complete finite language task."*

## Task

Dyck-1 / balanced parentheses, fixed length n; tokens `(`=0, `)`=1. Valid iff
running depth never negative and ends at 0 (genuinely requires the *order* check,
unlike a parenthesis count). Chosen over the saturated modular-addition and over
Gross's Max-of-K. Spec: `vcirc/dyck.py`.

## Architecture (and why)

`TinyTransformer` (`vcirc/model.py`): causal attention (→ per-position prefix
depth), ≥2 layers (depth → violation flags), sum-pool (→ final depth + violation
count), MLP readout (→ non-monotone `|final depth| == 0` test). No LayerNorm, so
later exact / rational re-evaluation stays clean.

## Three artefacts, three trust levels

1. **Lean theorem (kernel-checked):** `∀ x : Input n, Circuit.eval x = Spec.eval x`,
   proved by induction (not enumeration). Trust = the Lean kernel only.
2. **Exhaustive certificate (re-checkable):** for every input, the trained
   (rounded) model's argmax equals the circuit, with a positive logit margin;
   exact rational evaluation; a tiny standalone checker re-verifies from weights.
3. **Bridge (optional v2):** interval-arithmetic proof that the original float
   model preserves the rounded model's decisions.

## Milestones

- **A1 ✅** exact 100% full-domain model with positive margins (see PROGRESS.md).
- **A2** circuit extraction + exhaustive `Circuit == Model` certificate.
- **B** Lean proof `Circuit == Spec`.
- Scale to n=16 (65,536 inputs; likely minibatched training on CPU).

## Kill criteria (carried from the foothold)

Abandon/pivot if: no small circuit matches the model over the full domain;
agreement needs a brittle lookup-table rather than an algorithm; quantization
destroys margins; or the Lean proof devolves into enumeration. Backup tasks:
binary addition with carry; small-DFA recognition.
