"""Torch CRNN backend for local evaluation and the integration pipeline.

Mirrors OnnxRecognizer but runs the .pt checkpoint on GPU/CPU. Not part of the served
runtime (serving uses ONNX + onnxruntime, no torch).
"""

from __future__ import annotations

from pathlib import Path

import torch

from app.decode import greedy_decode
from app.models.crnn import CRNN
from app.recognizer import preprocess_band


class TorchRecognizer:
    def __init__(self, ckpt: str | Path = "models/crnn.pt", device: str | None = None, temperature: float = 1.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CRNN().to(self.device).eval()
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.temperature = temperature

    @torch.no_grad()
    def __call__(self, gray_band):
        x = torch.from_numpy(preprocess_band(gray_band)).to(self.device)
        logits = self.model(x)[0].float().cpu().numpy()  # (T, N_LOGITS)
        return greedy_decode(logits, self.temperature)
