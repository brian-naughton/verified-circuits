"""Export a model's weights to a torch-free, exact, hex text file.

Each float32 weight is written with `float.hex()`, which round-trips the value
**exactly** (a float is a dyadic rational). The standalone checker
(`certificates/check.py`) reads this file with `float.fromhex` and reconstructs
the exact rationals using only the Python standard library — so verifying the
certificate needs neither torch nor the training/extraction code.

Run: python -m vcirc.export_weights --model models/dyck10_exact_seed0.pt
"""
import argparse
import json
import os

import torch

from vcirc.model import TinyTransformer

ROOT = os.path.dirname(os.path.dirname(__file__))
CERT_DIR = os.path.join(ROOT, "certificates")


def _to_hex(x):
    if isinstance(x, list):
        return [_to_hex(e) for e in x]
    return float(x).hex()


def export(model_path: str, out_path: str) -> str:
    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    model = TinyTransformer(**blob["cfg"])
    model.load_state_dict(blob["state_dict"])
    sd_hex = {k: _to_hex(v.tolist()) for k, v in model.state_dict().items()}
    out = {"cfg": blob["cfg"], "state_dict_hex": sd_hex,
           "note": "weights as float.hex(); float.fromhex restores them exactly"}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=0, sort_keys=True)
        f.write("\n")
    return out_path


def default_out(model_path: str) -> str:
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(CERT_DIR, f"{stem}.weights.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dyck10_exact_seed0.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model_path = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    out_path = args.out or default_out(model_path)
    export(model_path, out_path)
    print(f"wrote exact hex weights -> {os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    main()
