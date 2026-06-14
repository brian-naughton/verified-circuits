"""A committed exact checkpoint must classify the ENTIRE domain correctly with a
positive decision margin — the core verified-interpretability claim at the model
level (the circuit/spec halves come in A2/B).
"""
import glob
import os

import pytest
import torch

from vcirc import dyck
from vcirc.model import TinyTransformer

MODELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyTransformer(**blob["cfg"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def _exact_models():
    return sorted(glob.glob(os.path.join(MODELS, "dyck*_exact_*.pt")))


@pytest.mark.parametrize("path", _exact_models())
def test_saved_model_is_exact_over_full_domain(path):
    model = _load(path)
    n = model.cfg["n"]
    strings = list(dyck.enumerate_all(n))
    X = torch.tensor(strings, dtype=torch.long)
    y = torch.tensor([int(dyck.is_valid(s)) for s in strings], dtype=torch.long)
    with torch.no_grad():
        logits = model(X)
    pred = logits.argmax(1)
    assert (pred == y).all(), f"{path}: not exact over the full domain"
    margin = (logits.gather(1, y[:, None]).squeeze(1)
              - logits.gather(1, (1 - y)[:, None]).squeeze(1)).min().item()
    assert margin > 0, f"{path}: non-positive min margin {margin}"


def test_there_is_at_least_one_exact_model():
    assert _exact_models(), "no exact checkpoint in models/ — run `python -m vcirc.train`"
