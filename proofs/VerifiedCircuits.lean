/-
  VerifiedCircuits — Milestone B root module.

  Imports the Dyck-1 spec, the extracted circuit, and the equivalence proof, and
  runs a faithfulness cross-check: that the Lean `Spec` reproduces the Catalan
  validity profile (so the translation of `vcirc/dyck.py` is faithful, not just
  internally consistent), and that circuit and spec agree on every short string
  (a finite corroboration of the proven, length-generic `circuit_eq_spec`).

  Lean core only — no Mathlib, no Batteries.
-/

import VerifiedCircuits.Basic
import VerifiedCircuits.Spec
import VerifiedCircuits.Circuit
import VerifiedCircuits.Equiv

namespace VerifiedCircuits

/-- All `2^n` length-`n` token strings. -/
def allStrings : Nat → List (List Tok)
  | 0 => [[]]
  | n + 1 =>
      (allStrings n).map (Tok.opn :: ·) ++ (allStrings n).map (Tok.cls :: ·)

/-- Number of valid Dyck-1 strings of length `n`. Equals `Catalan(n/2)` for even
    `n` and `0` for odd `n` — cf. `vcirc/dyck.py::count_valid`, which `tests/`
    verifies against the Catalan numbers. -/
def countValid (n : Nat) : Nat :=
  ((allStrings n).filter (fun s => Spec.isValid s)).length

-- Faithfulness: the Lean spec reproduces the Catalan profile.
-- Expect  [1, 0, 1, 0, 2, 0, 5, 0, 14, 0, 42]  for n = 0..10
--         (= C₀ C₁ C₂ C₃ C₄ C₅ interleaved with the odd-length zeros).
#eval (List.range 11).map countValid

-- Corroboration of `circuit_eq_spec` (which is *proved* for all lengths): the
-- circuit and the spec return the same Bool on every string up to length 12.
-- Expect  true.
#eval (List.range 13).all
  (fun n => (allStrings n).all (fun s => Circuit.valid s == Spec.isValid s))

-- The Milestone B artifact: kernel-checked, no `sorry`, no `native_decide`.
-- Expect ONLY  [propext, Quot.sound]  (no `sorryAx`, no `Lean.ofReduceBool`).
#print axioms circuit_eq_spec

end VerifiedCircuits
