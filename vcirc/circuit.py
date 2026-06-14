"""The extracted symbolic circuit — the program the trained model implements.

This is the human-readable algorithm the `TinyTransformer` was shown to compute,
established by the A2a probes (numbers below; see `docs/PROGRESS.md`). It is
written to mirror the model's *mechanism* — not merely its input/output — so that
each line corresponds to a measured internal computation:

    model stage            circuit step                       probe evidence
    ------------------     -------------------------------    ----------------------------
    causal attention   ->  per-position running depth d_i     linearly decodable per
      (prefix sum)                                            position, R^2 0.999-1.000
    per-position MLP   ->  per-position flag  v_i = [d_i<0]   perfectly linearly separable
      (threshold)                                             (LinearSVC 100%, both blocks)
    sum-pool           ->  final_depth     = d_n              R^2 0.99999, exact recovery
                           violation_count = sum_i v_i        R^2 0.991
    readout MLP        ->  valid := final_depth==0            predicted-valid == 1 iff both
                                    AND violation_count==0     zero; logit gap +10.4 there

Because ``violation_count == 0  <=>  min_prefix_depth >= 0`` (a prefix dips below
zero exactly when some ``d_i < 0``), this decision is precisely the Dyck-1
validity test of `vcirc/dyck.py`. The circuit therefore equals the spec *by
construction* — that is expected and is the easy half. The substance of
Milestone A2 is the exhaustively-checked certificate that the trained **model**
equals *this circuit* on every input (`vcirc/certify.py`); Milestone B is a Lean
proof that the circuit equals the spec.

Pure standard library — no torch, no numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from vcirc.dyck import String

# Token semantics, mirroring what the model's embedding sees:
#   0 = '(' -> depth +1     1 = ')' -> depth -1
OPEN: int = 0


def _increment(tok: int) -> int:
    """Depth increment the model accumulates for one token: '(' +1, ')' -1."""
    return 1 if tok == OPEN else -1


@dataclass(frozen=True)
class CircuitTrace:
    """The circuit's internal quantities for one input — the values the probes
    recovered from the model's activations.

    Attributes:
        depths: Running prefix depth ``d_i`` after each token (length n).
        violations: Per-position flag ``v_i = 1 iff d_i < 0`` (length n).
        final_depth: ``d_n`` — the aggregate read off the sum-pool.
        violation_count: ``sum_i v_i`` — the second aggregate from the sum-pool.
        valid: The decision, ``final_depth == 0 and violation_count == 0``.
    """

    depths: Tuple[int, ...]
    violations: Tuple[int, ...]
    final_depth: int
    violation_count: int
    valid: bool


class Circuit:
    """The symbolic circuit extracted from `models/dyck10_exact_seed0.pt`.

    `Circuit.eval` is the decision; `Circuit.trace` additionally exposes the
    per-position and aggregate intermediates so a checker (or reader) can see the
    mechanism, not just the answer.
    """

    @staticmethod
    def trace(x: String) -> CircuitTrace:
        """Run the circuit, returning all intermediate quantities.

        Args:
            x: A length-n input string as a tuple of tokens in ``{0, 1}``.

        Returns:
            A `CircuitTrace` with the per-position depths and violation flags,
            the two pooled aggregates, and the validity decision.
        """
        depth = 0
        depths = []
        violations = []
        violation_count = 0
        for tok in x:
            depth += _increment(tok)            # causal prefix sum  -> d_i
            depths.append(depth)
            v_i = 1 if depth < 0 else 0         # per-position threshold -> [d_i<0]
            violations.append(v_i)
            violation_count += v_i              # sum-pool aggregate
        final_depth = depth                     # sum-pool aggregate  -> d_n
        valid = (final_depth == 0) and (violation_count == 0)   # readout test
        return CircuitTrace(
            depths=tuple(depths),
            violations=tuple(violations),
            final_depth=final_depth,
            violation_count=violation_count,
            valid=valid,
        )

    @staticmethod
    def eval(x: String) -> bool:
        """The circuit's decision: True iff the model classifies ``x`` valid.

        Args:
            x: A length-n input string as a tuple of tokens in ``{0, 1}``.

        Returns:
            ``True`` iff ``final_depth == 0`` and no prefix went negative.
        """
        return Circuit.trace(x).valid


if __name__ == "__main__":
    # Sanity: the circuit equals the Dyck spec by construction, over the full
    # domain for several n (the genuinely-checked claim, model == circuit, is in
    # tests/test_certify.py).
    from vcirc import dyck

    for n in range(0, 15):
        ok = all(Circuit.eval(s) == dyck.is_valid(s) for s in dyck.enumerate_all(n))
        valid_n = sum(Circuit.eval(s) for s in dyck.enumerate_all(n))
        assert ok, f"circuit != spec at n={n}"
        print(f"n={n:2d}  circuit==spec over all {2**n:6d} inputs: {ok}  "
              f"(#valid={valid_n})")
