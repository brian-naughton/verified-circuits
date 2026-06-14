"""Certify `Circuit == Model` over the entire finite domain.

The claim certified, for a saved exact model and the extracted `Circuit`:

    for every input x in {0,1}^n, the model's argmax decision equals
    Circuit.eval(x), and the decision margin (logit[circuit] - logit[other]) is
    strictly positive.

Agreement on every input *with a strictly positive margin* is exactly
"model argmax == circuit, unambiguously, everywhere". Since `Circuit == Spec`
holds by construction (and is proved in Lean in Milestone B), this makes the
*model* provably exact on the whole domain — but the novelty is the mechanism +
exact checkability, not the accuracy number.

Rigour ladder (see `docs/A2-PLAN.md`):
  * v1 (this module, evidence): the deployed float32 model is run; we report the
    empirical agreement and min margin. The moment of truth — if any input
    disagrees, the extracted circuit is not faithful and we stop.
  * v2 (`vcirc/exact.py`, proof): exact-rational evaluation with a rigorous
    enclosure of the two softmaxes proves the margin is positive on every input,
    re-checkable by a torch-free standalone checker.

This module deliberately depends only on torch + the package (no training code).
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

from vcirc import dyck, exact, export_weights
from vcirc.circuit import Circuit
from vcirc.model import TinyTransformer

ROOT = os.path.dirname(os.path.dirname(__file__))
CERT_DIR = os.path.join(ROOT, "certificates")
CHECKER_VERSION = "a2b-v1-evidence"
CHECKER_VERSION_V2 = "a2b-v2-exact"


# --------------------------------------------------------------------------- #
# hashing helpers (so a certificate pins exactly what was checked)
# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# model + circuit evaluation over the full domain
# --------------------------------------------------------------------------- #
def load_model(path: str) -> TinyTransformer:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyTransformer(**blob["cfg"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def model_logits(model: TinyTransformer, strings: list) -> np.ndarray:
    """The deployed (float32) model's logits over the whole domain, as float64."""
    X = torch.tensor(strings, dtype=torch.long)
    with torch.no_grad():
        logits = model(X)
    return logits.numpy().astype(np.float64)


def circuit_decisions(strings: list) -> np.ndarray:
    """Circuit.eval over the whole domain as a 0/1 array (1 == valid)."""
    return np.array([int(Circuit.eval(s)) for s in strings], dtype=np.int64)


def verify(logits: np.ndarray, decision: np.ndarray, strings: list) -> dict:
    """Compare the model's argmax to the circuit's decision over the domain.

    The margin at x is ``logit[decision_x] - logit[1 - decision_x]`` — positive
    iff the model's argmax equals the circuit's decision strictly.

    Args:
        logits: (N, 2) model logits.
        decision: (N,) circuit decisions in {0, 1}.
        strings: the N domain inputs (to name the arg-min input).

    Returns:
        A dict with agreement flag, disagreement count, per-class confusion,
        the min/mean margin and the arg-min input.
    """
    n = len(decision)
    idx = np.arange(n)
    model_arg = logits.argmax(1)
    margin = logits[idx, decision] - logits[idx, 1 - decision]
    agree = model_arg == decision
    disagreements = int((~agree).sum())

    # confusion of circuit decision vs model argmax (should be perfectly diagonal)
    confusion = {
        "valid_valid": int(((decision == 1) & (model_arg == 1)).sum()),
        "valid_invalid": int(((decision == 1) & (model_arg == 0)).sum()),
        "invalid_valid": int(((decision == 0) & (model_arg == 1)).sum()),
        "invalid_invalid": int(((decision == 0) & (model_arg == 0)).sum()),
    }
    argmin = int(margin.argmin())
    return {
        "agreement": bool(agree.all()),
        "disagreements": disagreements,
        "per_class": {
            "circuit_valid": int((decision == 1).sum()),
            "circuit_invalid": int((decision == 0).sum()),
            "model_valid": int((model_arg == 1).sum()),
            "model_invalid": int((model_arg == 0).sum()),
            "confusion_circuit_x_model": confusion,
        },
        "min_margin": float(margin.min()),
        "mean_margin": float(margin.mean()),
        "min_margin_input": "".join(str(int(t)) for t in strings[argmin]),
    }


def build_certificate(model_path: str) -> dict:
    """Build (but do not write) the v1 certificate for a saved model."""
    model = load_model(model_path)
    n = model.cfg["n"]
    strings = list(dyck.enumerate_all(n))

    logits = model_logits(model, strings)
    decision = circuit_decisions(strings)
    result = verify(logits, decision, strings)

    cert = {
        "claim": ("for every input x in {0,1}^n, the model's argmax decision "
                  "equals Circuit.eval(x) with a strictly positive margin "
                  "(logit[circuit] - logit[other] > 0)"),
        "task": "dyck1",
        "n": n,
        "domain_size": len(strings),
        "rung": "v1-evidence",
        "arithmetic": "float32 (deployed torch model)",
        "model_file": os.path.relpath(model_path, ROOT),
        "model_sha256": sha256_file(model_path),
        "circuit_source": "vcirc/circuit.py",
        "circuit_sha256": sha256_file(os.path.join(ROOT, "vcirc", "circuit.py")),
        "spec_source": "vcirc/dyck.py",
        "spec_sha256": sha256_file(os.path.join(ROOT, "vcirc", "dyck.py")),
        "checker": "vcirc/certify.py",
        "checker_sha256": sha256_file(os.path.join(ROOT, "vcirc", "certify.py")),
        "checker_version": CHECKER_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **result,
    }
    return cert


def emit(cert: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cert, f, indent=2, sort_keys=True)
        f.write("\n")


def certify(model_path: str, out_dir: str = CERT_DIR, write: bool = True) -> dict:
    """Build, validate and (optionally) write the v1 certificate.

    Raises:
        AssertionError: if the model disagrees with the circuit on any input or
            the min margin is not strictly positive — the moment of truth.
    """
    cert = build_certificate(model_path)
    assert cert["agreement"] and cert["disagreements"] == 0, (
        f"MODEL != CIRCUIT on {cert['disagreements']} input(s) — the extracted "
        f"circuit is not faithful; stop and investigate before v2.")
    assert cert["min_margin"] > 0, (
        f"non-positive min margin {cert['min_margin']}")

    if write:
        stem = os.path.splitext(os.path.basename(model_path))[0]
        out_path = os.path.join(out_dir, f"{stem}.v1.cert.json")
        emit(cert, out_path)
        cert["_written_to"] = os.path.relpath(out_path, ROOT)
    return cert


# --------------------------------------------------------------------------- #
# v2 (proof): rigorous interval arithmetic on the exact-real function
# --------------------------------------------------------------------------- #
def build_certificate_v2(model_path: str, precision: int = 96,
                         jobs: int = 1) -> dict:
    """Build the v2 certificate: a rigorous lower bound on the decision margin.

    Exports the weights exactly, then evaluates the exact-real function defined
    by those weights in rigorous interval arithmetic (vcirc.exact). Every input's
    margin lower bound being positive *proves* the model's argmax equals the
    circuit on the whole domain — a portable, re-checkable guarantee.
    """
    weights_path = export_weights.default_out(model_path)
    export_weights.export(model_path, weights_path)

    with open(weights_path) as f:
        blob = json.load(f)
    cfg = blob["cfg"]
    sd = {k: exact._hex_to_float(v) for k, v in blob["state_dict_hex"].items()}
    n = cfg["n"]
    strings = list(dyck.enumerate_all(n))
    decision = circuit_decisions(strings)
    items = [(s, bool(decision[i])) for i, s in enumerate(strings)]

    res = exact.verify_domain(cfg, sd, items, precision=precision, jobs=jobs)
    margin = res["min_margin"]               # Fraction (rational lower bound)
    argmin = strings[res["argmin_index"]]

    cert = {
        "claim": (
            "the exact-real function defined by these weights has argmax == "
            "Circuit.eval(x) and a strictly positive decision margin (>= the "
            "stated rational lower bound) on every input x in {0,1}^n, proven by "
            "rigorous interval arithmetic [portable]; the deployed float32 model "
            "reproduces these decisions exhaustively over the full domain [v1]"),
        "task": "dyck1",
        "n": n,
        "domain_size": len(strings),
        "rung": "v2-exact",
        "arithmetic": (f"rigorous dyadic interval arithmetic, {precision}-bit "
                       f"precision, directed (outward) rounding; exp enclosed by "
                       f"Taylor+remainder; scale 1/sqrt(dh)={exact.Fraction(1, 4)} exact"),
        "argmax_equals_circuit_all_inputs": bool(res["all_positive"]),
        "min_margin_lower_bound": {
            "float": float(margin),
            "fraction": [margin.numerator, margin.denominator],
        },
        "min_margin_input": "".join(str(int(t)) for t in argmin),
        "interval_precision_bits": precision,
        "max_endpoint_bits": res["max_bits"],
        "per_class": {
            "circuit_valid": int((decision == 1).sum()),
            "circuit_invalid": int((decision == 0).sum()),
        },
        "model_file": os.path.relpath(model_path, ROOT),
        "model_sha256": sha256_file(model_path),
        "weights_export": os.path.relpath(weights_path, ROOT),
        "weights_export_sha256": sha256_file(weights_path),
        "circuit_source": "vcirc/circuit.py",
        "circuit_sha256": sha256_file(os.path.join(ROOT, "vcirc", "circuit.py")),
        "spec_source": "vcirc/dyck.py",
        "spec_sha256": sha256_file(os.path.join(ROOT, "vcirc", "dyck.py")),
        "verifier_core": "vcirc/exact.py",
        "verifier_core_sha256": sha256_file(os.path.join(ROOT, "vcirc", "exact.py")),
        "checker": "certificates/check.py",
        "checker_version": CHECKER_VERSION_V2,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return cert


def certify_v2(model_path: str, out_dir: str = CERT_DIR, precision: int = 96,
               jobs: int = 1, write: bool = True) -> dict:
    """Build, validate and (optionally) write the v2 certificate.

    Raises:
        AssertionError: if any input's margin lower bound is not strictly
            positive (the exact-real function would disagree with the circuit).
    """
    cert = build_certificate_v2(model_path, precision=precision, jobs=jobs)
    assert cert["argmax_equals_circuit_all_inputs"], (
        "exact-real function disagrees with the circuit on some input")
    assert cert["min_margin_lower_bound"]["float"] > 0, "non-positive margin bound"

    if write:
        stem = os.path.splitext(os.path.basename(model_path))[0]
        out_path = os.path.join(out_dir, f"{stem}.v2.cert.json")
        emit(cert, out_path)
        cert["_written_to"] = os.path.relpath(out_path, ROOT)
    return cert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dyck10_exact_seed0.pt")
    ap.add_argument("--rung", choices=["v1", "v2"], default="v1")
    ap.add_argument("--precision", type=int, default=96)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    model_path = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)

    if args.rung == "v1":
        cert = certify(model_path, write=not args.no_write)
        print(f"task=dyck1  n={cert['n']}  domain={cert['domain_size']}  rung={cert['rung']}")
        print(f"agreement: {cert['agreement']}   disagreements: {cert['disagreements']}")
        pc = cert["per_class"]
        print(f"circuit valid/invalid: {pc['circuit_valid']}/{pc['circuit_invalid']}   "
              f"confusion(circuit x model): {pc['confusion_circuit_x_model']}")
        print(f"min margin: {cert['min_margin']:+.4f} (at input {cert['min_margin_input']})   "
              f"mean margin: {cert['mean_margin']:+.4f}")
    else:
        cert = certify_v2(model_path, precision=args.precision, jobs=args.jobs,
                          write=not args.no_write)
        mm = cert["min_margin_lower_bound"]
        print(f"task=dyck1  n={cert['n']}  domain={cert['domain_size']}  rung={cert['rung']}")
        print(f"argmax == circuit on all inputs: {cert['argmax_equals_circuit_all_inputs']}")
        print(f"rigorous min margin lower bound: {mm['float']:+.6f}  "
              f"(= {mm['fraction'][0]}/{mm['fraction'][1]}) at input {cert['min_margin_input']}")
        print(f"precision: {cert['interval_precision_bits']} bits   "
              f"max endpoint bits: {cert['max_endpoint_bits']}")
    if "_written_to" in cert:
        print(f"wrote -> {cert['_written_to']}")


if __name__ == "__main__":
    main()
