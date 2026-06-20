"""Train the CRNN+CTC recognizer from scratch.

CTC runs in fp32 even under AMP (the loss is unstable in half), with zero_infinity to
survive the rare sample whose timestep count is shorter than its label. Best checkpoint is
chosen by validation exact-match. Run: python -m train.train_crnn
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.models.crnn import BLANK, CRNN
from eval.evaluate import evaluate_model
from train.data import MicrDataset, WidthBatchSampler, collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("models/crnn.pt"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = MicrDataset(args.data, "train")
    val_ds = MicrDataset(args.data, "val")
    sampler = WidthBatchSampler(train_ds.widths, args.batch, shuffle=True)
    loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers, pin_memory=True
    )

    model = CRNN().to(device)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print(f"train {len(train_ds)} | val {len(val_ds)} | batches/epoch {len(sampler)} | device {device}")
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        t0 = time.time()
        running = 0.0
        for x, targets, target_lengths, _labels, _tiers in loader:
            x = x.to(device, non_blocking=True)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)  # (B, T, C)
            logp = logits.float().log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((x.shape[0],), logp.shape[0], dtype=torch.long, device=device)
            loss = ctc(logp, targets, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()

        overall, _ = evaluate_model(model, val_ds, device, batch_size=args.batch * 2)
        s = overall.summary()
        print(
            f"epoch {epoch:2d} | loss {running / len(loader):.3f} | "
            f"val_exact {s['exact_match']:.4f} | val_cer {s['cer']:.4f} | {time.time() - t0:.0f}s"
        )
        if s["exact_match"] > best:
            best = s["exact_match"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "val_exact": best, "epoch": epoch}, args.out)
    print(f"best val exact-match: {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
