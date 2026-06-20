"""Fine-tune YOLO11n to detect the MICR band (single class).

Isolated training-only: ultralytics (AGPL-3.0) is imported ONLY here and in make_yolo_data,
never by the served app. The trained detector is exported to ONNX for the credential/blog
comparison against the classical localizer. Run: python -m train.yolo.train_yolo
"""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=_REPO / "datasets" / "yolo_micr" / "data.yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(_REPO / "runs"),
        name="micr_yolo",
        exist_ok=True,
        mosaic=0.0,  # a check is one scene; mosaic hurts and risks OOM on 4GB
        patience=12,
    )


if __name__ == "__main__":
    main()
