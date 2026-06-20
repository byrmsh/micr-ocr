"""Compare band detectors on the synthetic check val set: classical localizer vs YOLO11n.

Reports mean IoU and IoU@{0.5,0.7,0.9} for each, the "classical localization vs learned
detection" story for the blog. YOLO runs via ultralytics (training-only); the classical
localizer is the serving-path detector. Run: python -m eval.detect_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.localize import localize_band

_REPO = Path(__file__).resolve().parents[1]


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (aw * ah + bw * bh - inter + 1e-6)


def _gt_bbox(label_path: Path, w: int, h: int):
    cls, cx, cy, nw, nh = map(float, label_path.read_text().split())
    return (int((cx - nw / 2) * w), int((cy - nh / 2) * h), int(nw * w), int(nh * h))


def _summary(name: str, ious: list[float]) -> dict:
    a = np.array(ious)
    s = {
        "detector": name,
        "n": len(a),
        "mean_iou": round(float(a.mean()), 4),
        "iou@0.5": round(float((a >= 0.5).mean()), 4),
        "iou@0.7": round(float((a >= 0.7).mean()), 4),
        "iou@0.9": round(float((a >= 0.9).mean()), 4),
    }
    print(f"{name:10} mean_IoU={s['mean_iou']:.3f}  @0.5={s['iou@0.5']:.3f}  "
          f"@0.7={s['iou@0.7']:.3f}  @0.9={s['iou@0.9']:.3f}  (n={s['n']})")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path, default=_REPO / "datasets" / "yolo_micr")
    ap.add_argument("--weights", type=Path, default=_REPO / "runs" / "micr_yolo" / "weights" / "best.pt")
    ap.add_argument("--limit", type=int, default=800)
    args = ap.parse_args()

    images = sorted((args.val / "images" / "val").glob("*.png"))[: args.limit]
    gts, grays = [], []
    for img_path in images:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        gts.append(_gt_bbox(args.val / "labels" / "val" / f"{img_path.stem}.txt", w, h))
        grays.append(gray)

    classical = [iou(localize_band(g)[1], gt) for g, gt in zip(grays, gts)]
    _summary("classical", classical)

    if args.weights.exists():
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
        yolo_ious = []
        for img_path, gt in zip(images, gts):
            r = model.predict(str(img_path), verbose=False)[0]
            if len(r.boxes) == 0:
                yolo_ious.append(0.0)
                continue
            best = r.boxes[int(r.boxes.conf.argmax())]
            x0, y0, x1, y1 = best.xyxy[0].tolist()
            yolo_ious.append(iou((x0, y0, x1 - x0, y1 - y0), gt))
        _summary("yolo11n", yolo_ious)
    else:
        print(f"(YOLO weights not found at {args.weights}; train first)")


if __name__ == "__main__":
    main()
