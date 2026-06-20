"""Build a YOLO-format detection dataset of synthetic checks (single class: micr_band).

Images and normalized bbox labels come from app.synth.check. A fraction of each split uses
the held-out generator (heavier degradation, unseen textures) so the detector sees variety.
Run: python -m train.yolo.make_yolo_data
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
from pathlib import Path

import cv2

from app.synth import check

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "datasets" / "yolo_micr"


def _seed(ns: str, i: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{ns}:{i}".encode()).digest()[:8], "big")


def _one(task: tuple[str, int, str, bool]) -> None:
    ns, idx, split, held = task
    img, (x, y, bw, bh), _ = check.compose_check(_seed(ns, idx), held_out=held)
    h, w = img.shape
    cx = min(1.0, max(0.0, (x + bw / 2) / w))
    cy = min(1.0, max(0.0, (y + bh / 2) / h))
    nw, nh = min(1.0, bw / w), min(1.0, bh / h)
    cv2.imwrite(str(OUT / "images" / split / f"{idx:06d}.png"), img)
    (OUT / "labels" / split / f"{idx:06d}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


def _gen(split: str, n: int, ns: str, held_frac: float, workers: int) -> None:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    tasks = [(ns, i, split, (_seed(ns, i) % 100) / 100.0 < held_frac) for i in range(n)]
    with mp.Pool(workers) as pool:
        pool.map(_one, tasks, chunksize=32)
    print(f"{split}: {n} checks")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--held-frac", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    args = ap.parse_args()

    _gen("train", args.train, "yolo_train", args.held_frac, args.workers)
    _gen("val", args.val, "yolo_val", args.held_frac, args.workers)
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: micr_band\n"
    )
    print(f"wrote {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
