"""The Dyck spec must match known mathematics (Catalan numbers)."""

from vcirc import dyck


def test_valid_counts_are_catalan():
    # # balanced strings of length 2k is the k-th Catalan number
    catalan = [1, 1, 2, 5, 14, 42, 132, 429, 1430]
    for n in range(0, 17, 2):
        assert dyck.count_valid(n) == catalan[n // 2]


def test_odd_lengths_have_no_balanced_strings():
    for n in (1, 3, 5, 7):
        assert dyck.count_valid(n) == 0


def test_order_matters():
    assert dyck.is_valid((0, 1)) and not dyck.is_valid((1, 0))           # ()  vs  )(
    assert dyck.is_valid((0, 0, 1, 1)) and not dyck.is_valid((0, 1, 1, 0))


def test_depth_helpers():
    s = (0, 0, 1, 0, 1, 1)  # (()())
    assert dyck.final_depth(s) == 0
    assert dyck.min_prefix_depth(s) == 0
    assert dyck.depth_profile(s) == [1, 2, 1, 2, 1, 0]
    assert dyck.min_prefix_depth((1, 0)) == -1  # )( dips negative
