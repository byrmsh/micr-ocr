"""Export the trained CRNN to ONNX for serving (logits only; CTC decode stays in Python).

Dynamic batch and width so any band aspect works. Uses the legacy tracing exporter (the
dynamo exporter is flakier with LSTMs) at opset 17 for broad onnxruntime compatibility.
Verifies the ONNX logits match torch on a sample before writing is considered done.
Run: python -m train.export_crnn_onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from app.decode import INPUT_H
from app.models.crnn import CRNN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("models/crnn.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/onnx/crnn.onnx"))
    args = ap.parse_args()

    model = CRNN().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    args.out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.rand(1, 1, INPUT_H, 512)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 3: "width"}, "logits": {0: "batch", 1: "time"}},
        opset_version=17,
    )

    # Parity check: ONNX vs torch on a fresh random input of a different width.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    x = torch.rand(1, 1, INPUT_H, 736)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    print(f"exported {args.out} | onnx-vs-torch max|diff| = {max_diff:.2e}")
    if max_diff > 1e-3:
        raise SystemExit(f"ONNX parity check failed (max diff {max_diff:.2e})")


if __name__ == "__main__":
    main()
