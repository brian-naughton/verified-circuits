"""The v1 certificate is the moment of truth for `Circuit == Model`: the deployed
model's argmax must equal `Circuit.eval` on EVERY input with a strictly positive
margin. If this fails, the extracted circuit is not faithful.
"""
import glob
import json
import os

import numpy as np
import pytest
import torch

from vcirc import certify, dyck
from vcirc.circuit import Circuit

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(ROOT, "models")


def _exact_models():
    return sorted(glob.glob(os.path.join(MODELS, "dyck*_exact_*.pt")))


@pytest.mark.parametrize("model_path", _exact_models())
def test_model_argmax_equals_circuit_on_full_domain(model_path):
    # Direct check, independent of certify(): the central A2 claim.
    model = certify.load_model(model_path)
    n = model.cfg["n"]
    strings = list(dyck.enumerate_all(n))
    logits = certify.model_logits(model, strings)
    model_arg = logits.argmax(1)
    circ = np.array([int(Circuit.eval(s)) for s in strings])
    assert (model_arg == circ).all(), f"{model_path}: model != circuit somewhere"


@pytest.mark.parametrize("model_path", _exact_models())
def test_certificate_is_valid_and_consistent(model_path):
    cert = certify.certify(model_path, write=False)
    assert cert["agreement"] is True
    assert cert["disagreements"] == 0
    assert cert["min_margin"] > 0
    assert cert["domain_size"] == 2 ** cert["n"]
    # confusion must be perfectly diagonal (no off-diagonal disagreements)
    conf = cert["per_class"]["confusion_circuit_x_model"]
    assert conf["valid_invalid"] == 0 and conf["invalid_valid"] == 0
    # valid count must match the spec's Catalan count
    assert cert["per_class"]["circuit_valid"] == dyck.count_valid(cert["n"])
    assert cert["per_class"]["model_valid"] == cert["per_class"]["circuit_valid"]
    # the cert pins the artefacts it checked
    for k in ("model_sha256", "circuit_sha256", "checker_sha256"):
        assert isinstance(cert[k], str) and len(cert[k]) == 64


def test_certify_writes_loadable_json(tmp_path):
    model_path = _exact_models()[0]
    cert = certify.certify(model_path, out_dir=str(tmp_path), write=True)
    out = os.path.join(str(tmp_path), cert["_written_to"].split("/")[-1])
    assert os.path.exists(out)
    with open(out) as f:
        reloaded = json.load(f)
    assert reloaded["agreement"] is True
    assert reloaded["min_margin"] == cert["min_margin"]


def test_certify_rejects_a_tampered_circuit(monkeypatch):
    # If the circuit were wrong on even one input, certify() must refuse.
    model_path = _exact_models()[0]
    real_eval = Circuit.eval
    flipped = {"n": 0}

    def broken(s):  # flip the decision on exactly one input
        if not flipped["n"] and s == tuple([0, 1] * (len(s) // 2)):
            flipped["n"] = 1
            return not real_eval(s)
        return real_eval(s)

    monkeypatch.setattr(certify.Circuit, "eval", staticmethod(broken))
    with pytest.raises(AssertionError):
        certify.certify(model_path, write=False)


# --------------------------------------------------------------------------- #
# v2 (rigorous interval arithmetic)
# --------------------------------------------------------------------------- #
from vcirc import exact  # noqa: E402


def _weights_for(model_path):
    model = certify.load_model(model_path)
    sd = {k: v.tolist() for k, v in model.state_dict().items()}
    return exact.Weights(model.cfg, sd), model


@pytest.mark.parametrize("model_path", _exact_models())
def test_exact_interval_encloses_float64_and_margin_positive(model_path):
    # On a subset of the domain: the rigorous interval must enclose a float64
    # recompute, and the margin lower bound must be positive with the right sign.
    exact.set_precision(96)
    W, model = _weights_for(model_path)
    n = model.cfg["n"]
    strings = list(dyck.enumerate_all(n))
    mdouble = certify.load_model(model_path).double()
    # bounded spot-check: ~64 inputs spread across the domain, so the (heavy)
    # interval forward cost stays n-independent (at n=16, strings[::17] would be
    # ~3,855 inputs / ~24 min). The full-domain n=16 enclosure is established by
    # the v2 certificate + the torch-free re-check + experiments/validate_exact.py.
    step = max(1, len(strings) // 64)
    sub = strings[::step]
    Xs = torch.tensor(sub, dtype=torch.long)
    with torch.no_grad():
        L = mdouble(Xs).numpy()
    for k, s in enumerate(sub):
        g = exact.gap_interval(W, s)
        assert g.lo <= g.hi
        gap64 = float(L[k, 1] - L[k, 0])
        assert float(g.lo) - 1e-9 <= gap64 <= float(g.hi) + 1e-9
        mlo, _ = exact.margin_lower_bound(W, s, dyck.is_valid(s))
        assert mlo > 0


def test_exp_enclosure_is_rigorous():
    # exp bounds must bracket math.exp at a range of rational arguments.
    import math
    from fractions import Fraction
    exact.set_precision(96)
    for val in (-7.5, -1.0, 0.0, 0.3, 2.0, 9.25):
        X = int(Fraction(val) * exact._SCALE)
        lo_i, hi_i = exact.exp_bounds(X)
        lo, hi = lo_i / exact._SCALE, hi_i / exact._SCALE
        assert lo <= math.exp(val) <= hi
        assert hi - lo < 1e-12  # tight


def test_v2_certificate_file_is_consistent():
    # The committed v2 certificate must be internally consistent and pin a
    # weights export whose hash still matches.
    path = os.path.join(ROOT, "certificates", "dyck10_exact_seed0.v2.cert.json")
    if not os.path.exists(path):
        pytest.skip("v2 certificate not generated yet")
    with open(path) as f:
        cert = json.load(f)
    assert cert["rung"] == "v2-exact"
    assert cert["argmax_equals_circuit_all_inputs"] is True
    mm = cert["min_margin_lower_bound"]
    assert mm["float"] > 0
    from fractions import Fraction
    assert abs(float(Fraction(mm["fraction"][0], mm["fraction"][1])) - mm["float"]) < 1e-9
    wpath = os.path.join(ROOT, cert["weights_export"])
    assert certify.sha256_file(wpath) == cert["weights_export_sha256"]
    for k in ("model_sha256", "circuit_sha256", "verifier_core_sha256"):
        assert len(cert[k]) == 64
