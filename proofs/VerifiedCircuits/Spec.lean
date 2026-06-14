import VerifiedCircuits.Basic

/-
The Dyck-1 SPEC, a faithful image of `vcirc/dyck.py`.

  final_depth      (dyck.py:32-33)  ->  `finalDepth`
  min_prefix_depth (dyck.py:36-42)  ->  `minPrefixDepth` (via `minPrefixFrom`)
  is_valid         (dyck.py:45-47)  ->  `isValid`

`is_valid(s) = final_depth(s) == 0 and min_prefix_depth(s) >= 0`.
Lean core only.
-/

namespace VerifiedCircuits.Spec
open VerifiedCircuits

/-- Final depth = sum of per-token increments. Mirrors `dyck.py::final_depth`,
    `sum(step(t) for t in s)`. -/
def finalDepth : List Tok → Int
  | [] => 0
  | t :: ts => step t + finalDepth ts

/-- The running minimum of prefix depths, threaded with the running depth.
    `minPrefixFrom d m s` continues the loop of `dyck.py::min_prefix_depth` from
    current depth `d` and current running minimum `m`: it updates `d := d + step t`
    and `m := if d < m then d else m`, exactly as the Python `if d < m: m = d`. -/
def minPrefixFrom : Int → Int → List Tok → Int
  | _, m, [] => m
  | d, m, t :: ts =>
      minPrefixFrom (d + step t) (if d + step t < m then d + step t else m) ts

/-- Minimum prefix depth, including the empty prefix (Python inits `d = m = 0`).
    Mirrors `dyck.py::min_prefix_depth`. -/
def minPrefixDepth (s : List Tok) : Int := minPrefixFrom 0 0 s

/-- `s` is a valid Dyck-1 string: balanced (final depth 0) and no prefix goes
    negative. Mirrors `dyck.py::is_valid`,
    `final_depth(s) == 0 and min_prefix_depth(s) >= 0`. -/
def isValid (s : List Tok) : Bool :=
  decide (finalDepth s = 0) && decide (0 ≤ minPrefixDepth s)

end VerifiedCircuits.Spec
