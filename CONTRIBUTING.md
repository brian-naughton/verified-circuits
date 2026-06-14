# Contributing notes

A few conventions that keep the reproducibility / trust chain intact.

## Cert-pinned models are read-only

Any checkpoint in `models/` that a certificate references is **effectively
read-only**: a v2 certificate (`certificates/*.v2.cert.json`) verifies against the
checkpoint's exact `SHA256`, and the standalone `certificates/check.py` re-checks
that hash. Overwriting such a file silently breaks the published chain (the cert
would no longer match its model).

- Don't retrain over an existing `models/dyckN_exact_seed*.pt`. `vcirc/train.py`
  guards this: if the target already exists it saves to `models/scratch/`
  (gitignored) instead, unless you pass `--force` to overwrite intentionally.
- Smoke tests / sweeps / scratch experiments should write under `models/scratch/`.

## Trust surface

The two checkable halves are meant to be auditable by a stranger:

- `certificates/check.py` re-verifies a v2 certificate trusting only the Python
  stdlib + `vcirc/exact.py`. Its **default is a single-threaded, trivially
  auditable loop**; `--jobs=N` is an optional accelerator that runs the *same*
  exact-integer core across processes (bit-identical, order-independent) — use it
  for the large n=16 domain.
- `proofs/` (`cd proofs && lake build`) re-checks the Lean `Circuit == Spec`
  theorem trusting only the Lean kernel (Lean core only, no Mathlib).

Keep both paths torch-free and dependency-light.
