# proofs/ — Lean 4 (Milestone B)

Scaffold for the **kernel-checked** half of the guarantee.

Target theorem:

```lean
theorem dyck_circuit_correct (x : Input n) : Circuit.eval x = Spec.eval x
```

proved by induction over the input (not by enumerating all 2^n cases). When this
lands, a third party verifies `Circuit == Spec` by trusting only the Lean kernel:

```bash
lake build
```

Not yet started. Requires Lean 4 + Mathlib (toolchain via `elan`). The `Spec`
mirrors `vcirc/dyck.py`; the `Circuit` mirrors the symbolic circuit extracted in
Milestone A2 (`../certificates/`).
