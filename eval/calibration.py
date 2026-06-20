"""Confidence calibration + human-review routing.

Fits a single temperature on the CTC max-softmax confidence so it better matches the
probability that a read is fully correct, then derives an operating threshold for a target
auto-accept accuracy. We claim the METHOD, not a transferable threshold: the threshold is
fit on a held-out generator config and must be re-calibrated on a client's real
distribution. The blog states the CTC caveat (peaky, blank-dominated softmax) explicitly.

Outputs models/calibration.json (temperature, threshold, ECE before/after, coverage) and a
reliability + coverage-accuracy figure for the writeup. Run: python -m eval.calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.decode import emitted_logits, seq_conf_at_temp
from app.models.crnn import CRNN
from train.data import MicrDataset, WidthBatchSampler, collate


@torch.no_grad()
def collect(model, dataset, device: str) -> list[tuple[np.ndarray, bool]]:
    model.eval()
    sampler = WidthBatchSampler(dataset.widths, 128, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=4)
    out: list[tuple[np.ndarray, bool]] = []
    for x, _t, _tl, labels, _tiers in loader:
        logits = model(x.to(device)).float().cpu().numpy()
        for i, gt in enumerate(labels):
            text, rows = emitted_logits(logits[i])
            out.append((rows, text == gt))
    return out


def _bce(confs: np.ndarray, correct: np.ndarray) -> float:
    p = np.clip(confs, 1e-6, 1 - 1e-6)
    return float(-(correct * np.log(p) + (1 - correct) * np.log(1 - p)).mean())


def fit_temperature(samples: list[tuple[np.ndarray, bool]]) -> float:
    correct = np.array([c for _, c in samples], np.float64)
    best_t, best_loss = 1.0, np.inf
    for t in np.linspace(0.3, 6.0, 58):
        confs = np.array([seq_conf_at_temp(r, t) for r, _ in samples])
        loss = _bce(confs, correct)
        if loss < best_loss:
            best_loss, best_t = loss, float(t)
    return best_t


def ece(confs: np.ndarray, correct: np.ndarray, bins: int = 12) -> tuple[float, list]:
    edges = np.linspace(0, 1, bins + 1)
    total = len(confs)
    e = 0.0
    diagram = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs >= lo) & (confs < hi if hi < 1 else confs <= hi)
        if m.sum() == 0:
            continue
        acc = float(correct[m].mean())
        conf = float(confs[m].mean())
        e += (m.sum() / total) * abs(acc - conf)
        diagram.append({"conf": conf, "acc": acc, "n": int(m.sum())})
    return e, diagram


def coverage_accuracy(confs: np.ndarray, correct: np.ndarray) -> list[dict]:
    curve = []
    for thr in np.linspace(0, 1, 51):
        accepted = confs >= thr
        cov = float(accepted.mean())
        acc = float(correct[accepted].mean()) if accepted.sum() else 1.0
        curve.append({"threshold": float(thr), "coverage": cov, "accuracy": acc})
    return curve


def pick_threshold(curve: list[dict], target_acc: float) -> dict:
    feasible = [p for p in curve if p["accuracy"] >= target_acc and p["coverage"] > 0]
    return min(feasible, key=lambda p: p["threshold"]) if feasible else curve[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("models/crnn.pt"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--target-acc", type=float, default=0.99, help="strict near-certainty operating point")
    ap.add_argument("--serving-acc", type=float, default=0.975, help="balanced operating point the runtime uses")
    ap.add_argument("--out", type=Path, default=Path("models/calibration.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    samples = collect(model, MicrDataset(args.data, args.split), device)
    correct = np.array([c for _, c in samples])

    raw = np.array([seq_conf_at_temp(r, 1.0) for r, _ in samples])
    temperature = fit_temperature(samples)
    cal = np.array([seq_conf_at_temp(r, temperature) for r, _ in samples])

    ece_raw, _ = ece(raw, correct)
    ece_cal, reliability = ece(cal, correct)
    curve = coverage_accuracy(cal, correct)
    strict = pick_threshold(curve, args.target_acc)
    serving = pick_threshold(curve, args.serving_acc)

    result = {
        "split": args.split,
        "n": len(samples),
        "base_exact_match": float(correct.mean()),
        "temperature": temperature,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        # `serving_*` is the operating point the runtime applies (balanced auto-accept vs review).
        # `threshold` is the strict near-certainty point, kept for the writeup. Both are on the
        # temperature-scaled confidence; the recognizer applies `temperature` so serving thresholds
        # in the same space these were picked in.
        "serving_target_accuracy": args.serving_acc,
        "serving_threshold": serving["threshold"],
        "serving_coverage": serving["coverage"],
        "serving_accuracy": serving["accuracy"],
        "target_accuracy": args.target_acc,
        "threshold": strict["threshold"],
        "coverage_at_threshold": strict["coverage"],
        "accuracy_at_threshold": strict["accuracy"],
        "reliability": reliability,
        "coverage_accuracy": curve,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(
        f"T={temperature:.2f} | ECE {ece_raw:.3f}->{ece_cal:.3f} | "
        f"serving route<{serving['threshold']:.2f}: auto-accept {serving['coverage']:.1%} at "
        f"{serving['accuracy']:.1%} (target {args.serving_acc:.0%}) | "
        f"strict route<{strict['threshold']:.2f}: {strict['coverage']:.1%} at {strict['accuracy']:.1%} "
        f"(target {args.target_acc:.0%}) -> {args.out}"
    )
    _plot(reliability, curve, args.out.with_suffix(".png"))


def _plot(reliability: list, curve: list, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    rc = [d["conf"] for d in reliability]
    ra = [d["acc"] for d in reliability]
    a.plot([0, 1], [0, 1], "--", color="gray")
    a.plot(rc, ra, "o-")
    a.set(title="Reliability (calibrated)", xlabel="confidence", ylabel="accuracy")
    b.plot([p["coverage"] for p in curve], [p["accuracy"] for p in curve], "o-")
    b.set(title="Coverage vs accuracy", xlabel="coverage (auto-accepted)", ylabel="accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
