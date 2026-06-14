# proofs/ — Lean 4 (Milestone B)

The **kernel-checked** half of the guarantee: a Lean 4 proof that the extracted
circuit equals the Dyck-1 spec on every input.

## Theorem

```lean
theorem circuit_eq_spec (s : List Tok) : Circuit.valid s = Spec.isValid s
```

Proved by **structural induction over the input** (not by enumerating 2ⁿ cases),
so it holds for **every** length n — the headline n=16 domain and all others. With
the A2 `Circuit == Model` certificate this completes `Spec == Circuit == Model`.

## Reproduce

```bash
cd proofs
lake build      # green = the proof checks
```

The first build installs Lean 4.30.0 via the pinned `lean-toolchain` (through
`elan`). The build prints the faithfulness cross-checks and the axiom audit:

- `[1, 0, 1, 0, 2, 0, 5, 0, 14, 0, 42]` — the Lean spec reproduces the Catalan
  validity profile (n = 0..10), so it is a faithful image of `vcirc/dyck.py`.
- `true` — circuit and spec agree on every string up to length 12 (a finite
  corroboration of the proven, length-generic theorem).
- `'VerifiedCircuits.circuit_eq_spec' depends on axioms: [propext, Quot.sound]`.

## What you trust

Only the **Lean kernel** (plus `propext` and `Quot.sound`, the two standard axioms
the proof reports). No `sorry`; no `native_decide` (hence no `Lean.ofReduceBool`
compiler trust); **no Mathlib** — the whole project is Lean core only, four small
modules.

The Mathlib-free choice is about **build speed, reproducibility, and a small
auditable artifact — not soundness.** Lean's kernel checks Mathlib developments
just as rigorously; Mathlib is simply heavier. A reviewer can `lake build` this in
seconds.

## Scope

The domain is `List Tok` with `Tok = {opn, cls}` — exactly the model's binary
token alphabet (`(` = 0, `)` = 1, the only inputs the trained transformer ever
sees). The theorem is about that alphabet; it deliberately does not concern the
Python's incidental behaviour on out-of-alphabet integers (where `dyck.py`/
`circuit.py` would treat any non-zero token as `)`), which is outside the Dyck-1
domain and never reached.

## Layout

```
VerifiedCircuits/Basic.lean    -- Tok, step           (token encoding, +1 / -1)
VerifiedCircuits/Spec.lean     -- mirrors vcirc/dyck.py     (isValid)
VerifiedCircuits/Circuit.lean  -- mirrors vcirc/circuit.py  (valid)
VerifiedCircuits/Equiv.lean    -- lemmas + theorem circuit_eq_spec
VerifiedCircuits.lean          -- root: imports + #eval cross-checks + axiom audit
```

Each Lean definition cites the exact Python lines it mirrors.
