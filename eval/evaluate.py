"""Run a recognizer over a split and report CER / exact-match / per-field accuracy by tier.

Used both as a CLI (python -m eval.evaluate --model ... --split heldout) and as a library
(train loop calls evaluate_model for val exact-match). Ground-truth fields are parsed from
the well-formed gt label, so per-field accuracy needs nothing beyond the label.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.models.crnn import CRNN, greedy_decode
from app.synth.micr import parse_fields
from eval.metrics import Accumulator, format_table
from train.data import MicrDataset, WidthBatchSampler, collate


@torch.no_grad()
def evaluate_model(model, dataset: MicrDataset, device: str, batch_size: int = 128):
    model.eval()
    sampler = WidthBatchSampler(dataset.widths, batch_size, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=4)
    overall = Accumulator()
    by_tier: dict[str, Accumulator] = {}
    for x, _targets, _tl, labels, tiers in loader:
        logits = model(x.to(device)).float().cpu().numpy()  # (B, T, C)
        for i, gt in enumerate(labels):
            pred, _, _ = greedy_decode(logits[i])
            gt_fields = parse_fields(gt)
            overall.add(pred, gt, gt_fields)
            by_tier.setdefault(tiers[i], Accumulator()).add(pred, gt, gt_fields)
    return overall, by_tier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/crnn.pt"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--split", default="heldout")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRNN().to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt["model"])

    ds = MicrDataset(args.data, args.split)
    overall, by_tier = evaluate_model(model, ds, device)
    rows = {"OVERALL": overall.summary()}
    rows.update({t: by_tier[t].summary() for t in sorted(by_tier)})
    print(f"\n=== {args.split} (model: {args.model}) ===")
    print(format_table(rows))


if __name__ == "__main__":
    main()
