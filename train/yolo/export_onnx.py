"""Export the fine-tuned YOLO11n band detector to ONNX (for the blog comparison and an
optional learned-detector serving path). ultralytics is imported here only; the export is
a plain ONNX graph runnable with onnxruntime alone. Run: python -m train.yolo.export_onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=_REPO / "runs" / "micr_yolo" / "weights" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--out", type=Path, default=_REPO / "models" / "onnx" / "yolo_micr.onnx")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=17, dynamic=False, simplify=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, args.out)
    print(f"exported {onnx_path} -> {args.out}")


if __name__ == "__main__":
    main()
