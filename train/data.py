"""Torch dataset + width-bucketed batching for the recognizer.

Band crops are stored at STORE_H; here they are resized to the model's INPUT_H (keeping
aspect, variable width). Batches are bucketed by width so padding, and therefore the peak
activation memory that matters on a 4GB GPU, stays small.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from app.models.crnn import INPUT_H
from app.synth.dataset import STORE_H
from app.synth.glyphs import CHAR_TO_IDX


class MicrDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str, input_h: int = INPUT_H):
        self.dir = Path(data_dir)
        self.h = input_h
        with (self.dir / f"{split}.jsonl").open() as f:
            self.items = [json.loads(line) for line in f]
        scale = input_h / STORE_H
        self.widths = [max(input_h, round(it["w"] * scale)) for it in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        img = Image.open(self.dir / it["file"]).convert("L")
        w = max(self.h, round(img.width * self.h / img.height))
        img = img.resize((w, self.h), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0)[None]  # (1,H,W), white=1
        target = torch.tensor([CHAR_TO_IDX[c] for c in it["label"]], dtype=torch.long)
        return x, target, it["label"], it["tier"]


def collate(batch):
    imgs, targets, labels, tiers = zip(*batch)
    maxw = max(im.shape[2] for im in imgs)
    x = torch.ones(len(imgs), 1, imgs[0].shape[1], maxw)  # pad with white (1.0)
    for i, im in enumerate(imgs):
        x[i, :, :, : im.shape[2]] = im
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_cat = torch.cat(targets)
    return x, targets_cat, target_lengths, list(labels), list(tiers)


class WidthBatchSampler(Sampler):
    """Group width-sorted indices into batches; shuffle batch order each epoch."""

    def __init__(self, widths: list[int], batch_size: int, shuffle: bool = True, seed: int = 0):
        self.batches = []
        order = sorted(range(len(widths)), key=lambda i: widths[i])
        for k in range(0, len(order), batch_size):
            self.batches.append(order[k : k + batch_size])
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            order = [order[i] for i in torch.randperm(len(order), generator=g)]
        for b in order:
            yield self.batches[b]

    def __len__(self) -> int:
        return len(self.batches)
