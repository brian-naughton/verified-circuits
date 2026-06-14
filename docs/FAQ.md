# Reviewer FAQ

A skeptic's guide to what this repo proves, what it only corroborates, and how to
check it yourself. The guiding principle is **no overreach**: we separate the
claims by their trust level and say plainly which is which.

## What is this?

A tiny transformer is trained to *exact 100%* on a complete finite task —
**Dyck-1** (balanced parentheses) — over its entire input domain. We reverse-
engineer it into a human-readable symbolic **circuit** and ship end-to-end,
independently-checkable evidence that

> **Spec == Circuit == Model**  over *every* input,

with a kernel-checked proof of the circuit↔spec half.

## What is *proved* vs *corroborated*?

The end-to-end claim is **three links at three different trust levels**. "Kernel-
checked" applies to exactly one of them.

| Link | How established | Trust level | Scope |
|---|---|---|---|
| `Circuit == Spec` | Lean 4 theorem, by structural induction | **kernel-proven** (Lean kernel + `propext`, `Quot.sound`) | **all lengths** |
| `Circuit == Model` | rigorous interval-arithmetic certificate | machine-checked rational lower bound (stdlib + ~390-line interval core) | per-input, **n=10 and n=16** |
| Lean/Python defs ↔ same algorithm | faithful transcription | **corroborated, not proven** | binary `{(, )}` alphabet |

### 1. `Circuit == Spec` — kernel-proven, all lengths
`theorem circuit_eq_spec (s : List Tok) : Circuit.valid s = Spec.isValid s`, in
`proofs/`, proved by **induction over the input — not enumeration** — so it holds
for every length (n=16 and beyond) at once. `#print axioms circuit_eq_spec` reports
only `propext, Quot.sound`: no `sorry`, no `native_decide` (hence no
`Lean.ofReduceBool` compiler trust), no Mathlib. Trust = the Lean kernel.

### 2. `Circuit == Model` — rigorous certificate, n=10 and n=16
For every input, the *exact-real function* defined by the float32 weights has
`argmax == Circuit.eval(x)` with a **strictly positive** decision margin, proved by
rigorous interval arithmetic (rational endpoints, directed/outward rounding; the
only transcendental, `exp` in the two attention softmaxes, enclosed by a
Taylor+remainder bound; the scale `1/√dh = 1/4` is exact). Rational margin lower
bound ≥ **+6.3957** (n=10) and ≥ **+8.0950** (n=16, all 65,536 inputs). The
deployed *float32* execution is separately corroborated **exhaustively** over the
full domain (the v1 certificate). Trust = the Python stdlib + `vcirc/exact.py`.

### 3. The transcription link — corroborated, *not* proved
The Lean `Spec`/`Circuit` and the Python `dyck.py`/`circuit.py` must encode the
*same algorithm*. This link **cannot be formally proved** — Python is not a formal
object. We corroborate it four ways, and say so honestly:
1. the Lean spec reproduces the **Catalan numbers** `[1,0,1,0,2,0,5,0,14,0,42]`
   (n=0..10) — i.e. it is validated against *what Dyck-1 mathematically is*,
   independent of any `dyck.py` bug;
2. circuit == spec on every string up to length 12 (a finite cross-check of the
   proven theorem);
3. an independent line-by-line audit (an adversarial second model) against the
   cited Python line numbers;
4. every Lean definition cites the Python lines it mirrors.

## Three corroboration claims — don't conflate them

The interval certificate's **soundness is analytic**: every operation in
`vcirc/exact.py` rounds *outward*, so the computed interval rigorously encloses the
true real value by construction. Two empirical checks *corroborate* (do not
constitute) the guarantees, and target two different things:

1. **Enclosure-soundness corroboration** (is the interval bracketing the true
   exact-real value?): a high-precision float64 evaluation of the exact-real
   function lies inside the v2 interval, within float64's own error
   (`experiments/validate_exact.py`). Backup to the analytic guarantee — not the
   guarantee itself.
2. **float32-implementation faithfulness** (does the *deployed* run reproduce the
   exact decisions?): (a) decision agreement on every input — float32 `argmax` ==
   circuit == sign of the proven margin — and (b) numerical fidelity —
   `|float32 gap − exact gap|` within the float-error scale.

Note the v2 enclosure width (~1e-21) is far *tighter* than float32's own rounding
error (~1e-5), so a literal "float32 logit ∈ [lo, hi]" containment is **not**
expected and is not claimed; faithfulness is decision agreement + fidelity, above.

## How do I verify it myself?

```bash
# Circuit == Spec  (trust: the Lean kernel)
cd proofs && lake build          # green = proved; prints #print axioms

# Circuit == Model  (trust: Python stdlib + vcirc/exact.py; no torch)
python certificates/check.py certificates/dyck10_exact_seed0.v2.cert.json
python certificates/check.py certificates/dyck16_exact_seed0.v2.cert.json --jobs=8
```

The re-checker re-derives the circuit independently, recomputes every margin lower
bound from the weight export alone, and confirms the hash the certificate pinned.
Its default is a single-threaded, trivially auditable loop; `--jobs` is an optional
accelerator running the *same* exact-integer core (bit-identical, order-independent).

## How is this different from prior work?

- **Gross, Agrawal et al., _Compact Proofs of Model Performance via MI_ (NeurIPS
  2024).** Compact *formal performance lower bounds* via mechanistic
  interpretability (Max-of-K). The guarantee is an accuracy/loss bound.
- **Hadad, Katz, Bassan (ICLR 2026).** Verifier-certified circuit *robustness /
  minimality* (vision, continuous domains).
- **This repo.** Kernel-checked **circuit↔spec _equivalence_** for a *learned
  algorithm*, a total finite-domain `Spec == Circuit == Model`, shipped as a
  reproducible artifact a skeptic can re-verify without trusting our Python. The
  ownable phrase: *a kernel-checked mechanistic circuit for a learned transformer
  on a complete finite language task.*

## Honest limitations

- The model is **tiny** and the task is a **finite, simple** language. The novelty
  is the **mechanism + exact checkability** — an end-to-end equivalence proved/
  certified over the *whole* domain — **not** the accuracy number (100% on a small
  task is unremarkable on its own).
- The transcription link (Lean/Python ↔ same algorithm) is corroborated, not
  proved (see above); this is the one unavoidable soft seam.
- `Circuit == Model` is certified per-input at the scales shown (n=10, n=16);
  `Circuit == Spec` is the length-generic, all-inputs half.
- Scope is the model's binary `{(, )}` token alphabet (`proofs/README.md` §Scope).
