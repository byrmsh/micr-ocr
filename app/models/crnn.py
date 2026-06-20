"""CRNN + CTC recognizer (Shi et al. 2015, "An End-to-End Trainable Neural Network for
Image-based Sequence Recognition"), trained from scratch.

Channels are kept light (the real CRNN uses 512; 14 fixed-pitch glyphs need far less) so
training and inference fit a 4GB GPU. The network emits per-timestep logits over
NUM_CLASSES + 1 (the extra index is the CTC blank); CTC decoding and confidence
aggregation live outside the graph so only this logit-producing module is exported to ONNX.

Input: grayscale, fixed height INPUT_H, variable width. A 2x downsample twice gives
T = W/4 - 1 timesteps, comfortably longer than any MICR line.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from app.synth.glyphs import IDX_TO_CHAR, NUM_CLASSES

INPUT_H = 32
BLANK = NUM_CLASSES  # CTC blank index
N_LOGITS = NUM_CLASSES + 1


def _conv(i: int, o: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.ReLU(inplace=True)
    )


class CRNN(nn.Module):
    def __init__(self, n_logits: int = N_LOGITS, in_ch: int = 1, lstm_hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            _conv(in_ch, 32), nn.MaxPool2d(2, 2),  # 32 -> 16, W -> W/2
            _conv(32, 64), nn.MaxPool2d(2, 2),  # 16 -> 8, W/2 -> W/4
            _conv(64, 128), _conv(128, 128), nn.MaxPool2d((2, 1), (2, 1)),  # 8 -> 4
            _conv(128, 256), nn.MaxPool2d((2, 1), (2, 1)),  # 4 -> 2
            _conv(256, 256, k=2, p=0),  # 2 -> 1 (valid)
        )
        self.rnn = nn.LSTM(256, lstm_hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(lstm_hidden * 2, n_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)  # (B, 256, 1, T)
        b, c, h, w = f.shape
        if h != 1:
            raise ValueError(f"expected collapsed height 1, got {h}; input must be {INPUT_H}px tall")
        f = f.squeeze(2).permute(0, 2, 1)  # (B, T, 256)
        out, _ = self.rnn(f)
        return self.fc(out)  # (B, T, N_LOGITS) logits


def greedy_decode(logits: np.ndarray) -> tuple[str, list[float], float]:
    """CTC greedy decode of one sequence's logits (T, N_LOGITS).

    Returns (text, per_char_confidences, sequence_confidence). Per-char confidence is the
    softmax prob at the timestep that emitted the character; sequence confidence is the
    mean over emitted characters (1.0 for an empty read). Note the known CTC caveat: peaky,
    blank-dominated softmax makes these scores useful for *ranking* uncertainty, not as
    calibrated probabilities, until temperature-scaled (see eval/calibration).
    """
    probs = _softmax(logits)
    best = probs.argmax(axis=1)
    chars: list[str] = []
    confs: list[float] = []
    prev = BLANK
    for t, idx in enumerate(best):
        if idx != BLANK and idx != prev:
            chars.append(IDX_TO_CHAR[int(idx)])
            confs.append(float(probs[t, idx]))
        prev = idx
    seq_conf = float(np.mean(confs)) if confs else 1.0
    return "".join(chars), confs, seq_conf


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
