"""Generate band-crop datasets for the recognizer.

Each sample = one degraded MICR band (grayscale PNG, fixed height STORE_H, variable width)
plus its label and parsed fields. Seeds are derived deterministically from
(namespace, index), so a split is fully reproducible and the held-out family never collides
with the training family. Run as a module to materialize splits to data/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import degrade as dg
from . import micr, render
from .config import HELDOUT_CONFIG, TRAIN_CONFIG, GeneratorConfig

STORE_H = 48  # stored band-crop height; the recognizer resizes to its own input height
_REPO = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO / "data"


def _seed(namespace: str, idx: int) -> int:
    h = hashlib.sha256(f"{namespace}:{idx}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _paper(h: int, w: int, rng: np.random.Generator, tint_p: float) -> np.ndarray:
    """A near-white paper background with faint fiber noise and optional security mottling."""
    paper = np.full((h, w), 255.0, np.float32)
    paper -= np.abs(rng.normal(0, 4, (h, w)))  # fiber speckle
    if rng.random() < tint_p:  # low-frequency tint/mottle
        low = rng.normal(0, 1, (max(2, h // 8), max(2, w // 8))).astype(np.float32)
        mottle = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
        paper -= np.abs(cv2.GaussianBlur(mottle, (0, 0), 3)) * rng.uniform(6, 16)
    return paper.clip(200, 255)


def make_band(seed: int, tier_name: str, cfg: GeneratorConfig):
    """Build one degraded band image (uint8, STORE_H tall) and its MicrLine label."""
    rng = np.random.default_rng(seed)
    line = micr.random_micr_line(random.Random(seed))
    tier = dg.TIERS[tier_name]

    band_h = int(rng.integers(cfg.band_h[0], cfg.band_h[1] + 1))
    cell_w = render.glyphs.cell_width_px(band_h)
    pad_x = int(cell_w * rng.uniform(0.4, 0.9))
    pad_y = int(band_h * rng.uniform(0.3, 0.6))
    cov, _, _ = render.render_line(
        line.render_text, band_h, pad_x=pad_x, pad_y=pad_y, jitter=cfg.pitch_jitter, rng=rng
    )

    h, w = cov.shape
    paper = _paper(h, w, rng, cfg.tint_p)
    ink_value = float(rng.integers(cfg.ink_value[0], cfg.ink_value[1] + 1))
    img = (paper * (1.0 - cov) + ink_value * cov).clip(0, 255).astype(np.uint8)

    img = dg.degrade(img, tier, rng)
    if cfg.elastic:
        img = dg.elastic_warp(img, rng)

    scale = STORE_H / img.shape[0]
    img = cv2.resize(img, (max(1, round(img.shape[1] * scale)), STORE_H), interpolation=cv2.INTER_AREA)
    return img, line


def _pick_tier(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = list(weights)
    p = np.array([weights[n] for n in names], np.float64)
    return names[int(rng.choice(len(names), p=p / p.sum()))]


def _gen_one(task: tuple[str, int, GeneratorConfig, Path]) -> dict:
    namespace, idx, cfg, img_dir = task
    seed = _seed(namespace, idx)
    tier = _pick_tier(np.random.default_rng(seed ^ 0xABCD), cfg.tier_weights)
    img, line = make_band(seed, tier, cfg)
    name = f"{idx:06d}.png"
    Image.fromarray(img, mode="L").save(img_dir / name)
    return {
        "file": f"{namespace}/{name}",
        "label": line.label,
        "tier": tier,
        "w": int(img.shape[1]),
        "routing": line.routing,
        "account": line.account,
        "check_number": line.check_number,
        "amount": line.amount,
    }


def generate_split(out_dir: Path, namespace: str, n: int, cfg: GeneratorConfig, workers: int) -> None:
    img_dir = out_dir / namespace
    img_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(namespace, i, cfg, img_dir) for i in range(n)]
    if workers > 1:
        with mp.Pool(workers) as pool:
            records = pool.map(_gen_one, tasks, chunksize=64)
    else:
        records = [_gen_one(t) for t in tasks]
    labels_path = out_dir / f"{namespace}.jsonl"
    with labels_path.open("w") as f:
        for rec in records:  # records keep task order
            f.write(json.dumps(rec) + "\n")
    print(f"{namespace}: {n} samples -> {labels_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic E-13B band-crop datasets")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--heldout", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    args = ap.parse_args()

    generate_split(args.out, "train", args.train, TRAIN_CONFIG, args.workers)
    generate_split(args.out, "val", args.val, TRAIN_CONFIG, args.workers)
    generate_split(args.out, "test", args.test, TRAIN_CONFIG, args.workers)
    generate_split(args.out, "heldout", args.heldout, HELDOUT_CONFIG, args.workers)


if __name__ == "__main__":
    main()
