import VerifiedCircuits.Basic

/-
The extracted CIRCUIT, a faithful image of `vcirc/circuit.py::Circuit.trace` /
`Circuit.eval` (the symbolic program the trained TinyTransformer was shown to
compute; see docs/PROGRESS.md A2a).

  depth += _increment(tok); final_depth = depth        (circuit.py:91,96)
      ->  `finalDepth` (via `finalDepthFrom`)
  v_i = 1 if depth < 0 else 0; violation_count += v_i  (circuit.py:93,95)
      ->  `violationCount` (via `violationCountFrom`)
  valid = (final_depth == 0) and (violation_count == 0)  (circuit.py:97)
      ->  `valid`

Note the order, mirroring the Python loop body exactly: the increment is applied
first (`depth += _increment(tok)`), then the violation flag tests the NEW depth
(`depth < 0`).  Lean core only.
-/

namespace VerifiedCircuits.Circuit
open VerifiedCircuits

/-- The circuit's running depth accumulator `d_i` (causal prefix sum), continued
    from `d`. Mirrors `circuit.py`'s `depth += _increment(tok)` (line 91);
    `final_depth = depth` after the whole string (line 96). -/
def finalDepthFrom : Int → List Tok → Int
  | d, [] => d
  | d, t :: ts => finalDepthFrom (d + step t) ts

/-- Circuit final depth `d_n` — the first sum-pool aggregate. -/
def finalDepth (s : List Tok) : Int := finalDepthFrom 0 s

/-- The circuit's violation count `sum_i [d_i < 0]`, threaded with the running
    depth `d`. Mirrors `circuit.py`'s `v_i = 1 if depth < 0 else 0;
    violation_count += v_i` (lines 93, 95) — the test uses the post-increment
    depth `d + step t`. -/
def violationCountFrom : Int → List Tok → Int
  | _, [] => 0
  | d, t :: ts =>
      (if d + step t < 0 then 1 else 0) + violationCountFrom (d + step t) ts

/-- Circuit violation count — the second sum-pool aggregate. -/
def violationCount (s : List Tok) : Int := violationCountFrom 0 s

/-- The circuit's decision: valid iff final depth is 0 and there are no
    violations. Mirrors `circuit.py`'s
    `valid = (final_depth == 0) and (violation_count == 0)` (line 97). -/
def valid (s : List Tok) : Bool :=
  decide (finalDepth s = 0) && decide (violationCount s = 0)

end VerifiedCircuits.Circuit
