/-
Shared token type and per-token depth increment, used by both the Dyck-1 SPEC
(`VerifiedCircuits.Spec`, mirroring `vcirc/dyck.py`) and the extracted CIRCUIT
(`VerifiedCircuits.Circuit`, mirroring `vcirc/circuit.py`).

Token encoding mirrors the Python (`vcirc/dyck.py` lines 14, 18-20):
  Python  0 = '(' = depth +1   <->   `Tok.opn`
  Python  1 = ')' = depth -1   <->   `Tok.cls`
A length-n "string" is a `List Tok` (the Python `String = Tuple[int, ...]`).

Lean core only — no Mathlib, no Batteries.
-/

namespace VerifiedCircuits

/-- A bracket token. `opn` is '(' (Python encoding 0); `cls` is ')' (Python 1). -/
inductive Tok where
  | opn
  | cls
deriving DecidableEq, Repr

/-- Depth increment for one token: '(' -> +1, ')' -> -1.
    Mirrors `vcirc/dyck.py::step` (line 18-20) and `vcirc/circuit.py::_increment`
    (line 42-44). -/
def step : Tok → Int
  | .opn => 1
  | .cls => -1

end VerifiedCircuits
