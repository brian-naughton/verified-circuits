"""The extracted circuit must equal the Dyck spec over the whole domain (the
'circuit == spec' half, here checked by enumeration; Milestone B proves it in
Lean by induction) and must expose correct intermediate quantities.
"""

from vcirc import dyck
from vcirc.circuit import Circuit


def test_circuit_equals_spec_over_full_domain():
    for n in range(0, 15):
        for s in dyck.enumerate_all(n):
            assert Circuit.eval(s) == dyck.is_valid(s), f"mismatch at n={n}, s={s}"


def test_trace_intermediates_match_dyck_helpers():
    # The circuit's exposed intermediates are exactly the spec's quantities.
    for s in dyck.enumerate_all(8):
        tr = Circuit.trace(s)
        assert list(tr.depths) == dyck.depth_profile(s)
        assert tr.final_depth == dyck.final_depth(s)
        assert (tr.violation_count == 0) == (dyck.min_prefix_depth(s) >= 0)
        assert tr.valid == dyck.is_valid(s)


def test_violation_count_is_zero_iff_min_prefix_nonnegative():
    # The readout uses violation_count==0; this is the order check (min prefix).
    for s in dyck.enumerate_all(10):
        tr = Circuit.trace(s)
        assert (tr.violation_count == 0) == (dyck.min_prefix_depth(s) >= 0)


def test_known_examples():
    assert Circuit.eval((0, 1)) is True            # ()
    assert Circuit.eval((1, 0)) is False           # )(  dips negative
    assert Circuit.eval((0, 0, 1, 1)) is True      # (())
    assert Circuit.eval((0, 1, 1, 0)) is False     # ()) (  prefix goes negative
    tr = Circuit.trace((0, 1, 1, 0))
    assert tr.depths == (1, 0, -1, 0)
    assert tr.violations == (0, 0, 1, 0)
    assert tr.final_depth == 0 and tr.violation_count == 1 and tr.valid is False
