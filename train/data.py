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
    def __init__(self, data_dir: str | Path, split: str, input_h: int = INPUT_H, augment: bool = False):
        self.dir = Path(data_dir)
        self.h = input_h
        self.augment = augment
        with (self.dir / f"{split}.jsonl").open() as f:
            self.items = [json.loads(line) for line in f]
        scale = input_h / STORE_H
        self.widths = [max(input_h, round(it["w"] * scale)) for it in self.items]

    def _crop_jitter(self, arr: np.ndarray) -> np.ndarray:
        """Random crop between the ink bbox (tight) and the full image (loose), per side.

        Makes the recognizer robust to how much margin the detector leaves, without ever
        cutting a glyph. This is the main lever closing the band-only vs end-to-end gap.
        """
        ys, xs = np.where(arr < 200)
        if len(xs) == 0:
            return arr
        h, w = arr.shape
        x0i, x1i, y0i, y1i = xs.min(), xs.max(), ys.min(), ys.max()
        left = np.random.randint(0, x0i + 1)
        right = np.random.randint(x1i + 1, w + 1)
        top = np.random.randint(0, y0i + 1)
        bot = np.random.randint(y1i + 1, h + 1)
        return arr[top:bot, left:right]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        arr = np.asarray(Image.open(self.dir / it["file"]).convert("L"))
        if self.augment:
            arr = self._crop_jitter(arr)
        w = max(self.h, round(arr.shape[1] * self.h / arr.shape[0]))
        resized = Image.fromarray(arr).resize((w, self.h), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(resized, np.float32) / 255.0)[None]  # (1,H,W), white=1
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
