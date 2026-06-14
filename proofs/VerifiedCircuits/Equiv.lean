import VerifiedCircuits.Spec
import VerifiedCircuits.Circuit

/-
Milestone B theorem: the extracted circuit computes exactly the Dyck-1 spec, on
EVERY input of EVERY length — proved by structural induction over the input, NOT
by enumerating cases. (So it covers the headline n = 16 domain, and every other,
for free.)

Proof structure:
  * the two `final_depth`s agree                              (`finalDepth_eq`)
  * both order conditions reduce to "every prefix depth >= 0":
      violation_count = 0   <->  allNonneg   (`violationCountFrom_eq_zero_iff`)
      min_prefix_depth >= 0 <->  allNonneg   (`minPrefixFrom_nonneg_iff`)
  * combine                                                   (`circuit_eq_spec`)

Lean core + `omega` only.
-/

namespace VerifiedCircuits

/-- Starting from depth `d`, every running prefix depth `d + step t₁`,
    `d + step t₁ + step t₂`, … is `≥ 0`. This is the common core of both order
    conditions (the circuit's "no violations" and the spec's "min prefix ≥ 0"). -/
def allNonnegFrom : Int → List Tok → Prop
  | _, [] => True
  | d, t :: ts => 0 ≤ d + step t ∧ allNonnegFrom (d + step t) ts

namespace Circuit

/-- The violation count is a sum of 0/1 flags, hence never negative. -/
theorem violationCountFrom_nonneg (d : Int) (s : List Tok) :
    0 ≤ violationCountFrom d s := by
  induction s generalizing d with
  | nil => simp [violationCountFrom]
  | cons t ts ih =>
    simp only [violationCountFrom]
    have h2 := ih (d + step t)
    by_cases h : d + step t < 0
    · simp only [if_pos h]; omega
    · simp only [if_neg h]; omega

/-- The circuit reports no violations iff every running prefix depth is ≥ 0. -/
theorem violationCountFrom_eq_zero_iff (d : Int) (s : List Tok) :
    violationCountFrom d s = 0 ↔ allNonnegFrom d s := by
  induction s generalizing d with
  | nil => simp [violationCountFrom, allNonnegFrom]
  | cons t ts ih =>
    simp only [violationCountFrom, allNonnegFrom]
    have hnn := violationCountFrom_nonneg (d + step t) ts
    have ihe := ih (d + step t)
    by_cases h : d + step t < 0
    · simp only [if_pos h]
      constructor
      · intro hc; exfalso; omega
      · rintro ⟨hd, _⟩; exfalso; omega
    · simp only [if_neg h, Int.zero_add]
      rw [ihe]
      constructor
      · intro ha; exact ⟨by omega, ha⟩
      · rintro ⟨_, ha⟩; exact ha

end Circuit

namespace Spec

/-- The running minimum (from depth `d`, current min `m`) stays ≥ 0 iff the
    current min is ≥ 0 and every running prefix depth is ≥ 0. -/
theorem minPrefixFrom_nonneg_iff (d m : Int) (s : List Tok) :
    0 ≤ minPrefixFrom d m s ↔ (0 ≤ m ∧ allNonnegFrom d s) := by
  induction s generalizing d m with
  | nil => simp [minPrefixFrom, allNonnegFrom]
  | cons t ts ih =>
    simp only [minPrefixFrom, allNonnegFrom]
    rw [ih (d + step t) (if d + step t < m then d + step t else m)]
    by_cases h : d + step t < m
    · simp only [if_pos h]
      constructor
      · rintro ⟨hd, ha⟩; exact ⟨by omega, hd, ha⟩
      · rintro ⟨_, hd, ha⟩; exact ⟨hd, ha⟩
    · simp only [if_neg h]
      constructor
      · rintro ⟨hm, ha⟩; exact ⟨hm, by omega, ha⟩
      · rintro ⟨hm, _, ha⟩; exact ⟨hm, ha⟩

end Spec

/-- The circuit's accumulator depth equals the spec's summed depth (offset by the
    start depth). -/
theorem finalDepthFrom_eq (d : Int) (s : List Tok) :
    Circuit.finalDepthFrom d s = d + Spec.finalDepth s := by
  induction s generalizing d with
  | nil => simp [Circuit.finalDepthFrom, Spec.finalDepth]
  | cons t ts ih =>
    simp only [Circuit.finalDepthFrom, Spec.finalDepth]
    rw [ih (d + step t)]; omega

/-- Circuit and spec compute the same final depth. -/
theorem finalDepth_eq (s : List Tok) : Circuit.finalDepth s = Spec.finalDepth s := by
  simp only [Circuit.finalDepth]
  rw [finalDepthFrom_eq]; omega

/-- **Milestone B.** The extracted circuit equals the Dyck-1 spec on every input
    of every length, proved by structural induction (not enumeration). With the
    A2 `Circuit == Model` certificate this completes `Spec == Circuit == Model`. -/
theorem circuit_eq_spec (s : List Tok) : Circuit.valid s = Spec.isValid s := by
  have hord : (Circuit.violationCount s = 0) ↔ (0 ≤ Spec.minPrefixDepth s) := by
    unfold Circuit.violationCount Spec.minPrefixDepth
    rw [Circuit.violationCountFrom_eq_zero_iff, Spec.minPrefixFrom_nonneg_iff]
    constructor
    · intro h; exact ⟨by omega, h⟩
    · rintro ⟨_, h⟩; exact h
  have hd1 : decide (Circuit.finalDepth s = 0) = decide (Spec.finalDepth s = 0) := by
    rw [finalDepth_eq]
  have hd2 : decide (Circuit.violationCount s = 0) = decide (0 ≤ Spec.minPrefixDepth s) := by
    simp only [hord]
  unfold Circuit.valid Spec.isValid
  rw [hd1, hd2]

end VerifiedCircuits
