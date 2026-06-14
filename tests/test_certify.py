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
