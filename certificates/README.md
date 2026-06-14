# certificates/ — Circuit == Model (Milestone A2)

The **independently-checkable** half of the guarantee: for a saved exact model
and the extracted symbolic `Circuit`, A2 certifies that — for *every* input in
the finite domain — the model's argmax equals the circuit's decision with a
strictly positive margin.

We certify the **deployed model directly**: float32 weights are exact dyadic
rationals, so there is no rounded model to bridge to.

## Files

- `<model>.weights.json` — the model's weights as exact `float.hex()` strings
  (`float.fromhex` restores them with no rounding; torch-free).
- `<model>.v1.cert.json` — **evidence rung**: the deployed float32 model run over
  the whole domain; reports exhaustive argmax agreement + the empirical min
  margin. Produced by `python -m vcirc.certify --rung v1`.
- `<model>.v2.cert.json` — **proof rung** (the differentiator): the exact-real
  function defined by the weights, evaluated in rigorous interval arithmetic
  (rational endpoints, directed rounding; `exp` enclosed; scale `1/√dh = 1/4`
  exact). States a **rational lower bound on the decision margin** that is
  strictly positive on every input. Produced by
  `python -m vcirc.certify --rung v2 --jobs 4`.
- `check.py` — a tiny **standalone re-verifier** for a v2 certificate. It
  recomputes the weights' hash, reconstructs the exact rationals from the hex
  export, re-derives the circuit, and re-runs the interval forward to confirm
  every margin lower bound is positive — trusting only the Python standard
  library and `vcirc/exact.py` (no torch, no training/extraction code):

  ```bash
  python certificates/check.py certificates/dyck10_exact_seed0.v2.cert.json
  ```

## Current result (`dyck10_exact_seed0`, n=10, 1,024 inputs)

- argmax == circuit on **all** inputs (both rungs).
- rigorous min margin lower bound **≥ +6.3957** (exact rational; 96-bit interval
  precision, max endpoint size 105 bits — no bit-blowup).
