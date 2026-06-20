"""Benchmark the recognizer ladder across difficulty tiers and report the results table.

Three honest rungs:
  template  - non-ML normalized-cross-correlation matcher (the real classical approach)
  crnn      - the trained CRNN+CTC recognizer
  tesseract - stock OCR, a labeled "general OCR cannot read E-13B" reference (subsampled)

Each is run on band crops from a split, scored by tier. Output is printed and saved to
models/benchmark_<split>.json for the blog. Run: python -m eval.benchmark --split heldout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.synth.micr import parse_fields
from eval.metrics import Accumulator, format_table


def _iter_split(data_dir: Path, split: str, limit: int | None = None):
    items = [json.loads(line) for line in (data_dir / f"{split}.jsonl").open()]
    if limit:
        items = items[:limit]
    for it in items:
        gray = np.asarray(Image.open(data_dir / it["file"]).convert("L"))
        yield gray, it["label"], it["tier"]


def run_recognizer(recognizer, data_dir: Path, split: str, limit: int | None = None):
    overall = Accumulator()
    by_tier: dict[str, Accumulator] = {}
    for gray, gt, tier in _iter_split(data_dir, split, limit):
        pred, _, _ = recognizer(gray)
        gtf = parse_fields(gt)
        overall.add(pred, gt, gtf)
        by_tier.setdefault(tier, Accumulator()).add(pred, gt, gtf)
    return overall, by_tier


def _build(name: str):
    if name == "crnn":
        from eval.torch_recognizer import TorchRecognizer

        return TorchRecognizer()
    if name == "template":
        from eval.baselines import TemplateMatcher

        return TemplateMatcher()
    if name == "tesseract":
        from eval.baselines import TesseractBaseline

        return TesseractBaseline()
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--recognizers", nargs="+", default=["template", "crnn", "tesseract"])
    ap.add_argument("--tesseract-limit", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report: dict[str, dict] = {}
    for name in args.recognizers:
        rec = _build(name)
        limit = args.tesseract_limit if name == "tesseract" else None
        overall, by_tier = run_recognizer(rec, args.data, args.split, limit)
        rows = {"OVERALL": overall.summary()} | {t: by_tier[t].summary() for t in sorted(by_tier)}
        report[name] = rows
        print(f"\n### {name}  (split={args.split}{', limit=' + str(limit) if limit else ''})")
        print(format_table(rows))

    out = args.out or Path("models") / f"benchmark_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
