# certificates/ — Circuit == Model (Milestone A2)

Scaffold for the **exhaustively-checked** half of the guarantee.

For a saved exact model and an extracted symbolic circuit, A2 emits a certificate
asserting, for *every* input in the finite domain:

- the trained (rounded) model's argmax equals the circuit's output, and
- the winning logit has a positive margin (lower bound, exact arithmetic).

A certificate is JSON with: domain size, model + circuit hashes, per-input
agreement, the minimum margin, and the checker version. A tiny standalone checker
re-verifies it from the weights in exact (rational) arithmetic — independent of
the training/extraction code.

Not yet generated. Produced by `vcirc/certify.py` (to be added in A2).
