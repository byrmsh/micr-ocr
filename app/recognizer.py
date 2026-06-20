"""Serving-safe recognition pipeline: localize -> recognize -> confidence -> parse.

No torch here. The ONNX backend (onnxruntime) is what runs in the deployed container; a
torch backend used for local evaluation lives in eval/torch_recognizer.py and reuses the
same preprocess and decode. A recognizer is any callable: grayscale band -> (text,
per_char_confidences, sequence_confidence).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.decode import INPUT_H, greedy_decode
from app.localize import localize_band
from app.synth.micr import parse_fields

Recognizer = Callable[[np.ndarray], tuple[str, list[float], float]]


def preprocess_band(gray: np.ndarray) -> np.ndarray:
    """Grayscale band -> (1, 1, INPUT_H, W) float32 in [0,1], white = 1."""
    h, w = gray.shape
    new_w = max(INPUT_H, round(w * INPUT_H / h))
    resized = cv2.resize(gray, (new_w, INPUT_H), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[None, None]


@dataclass
class MicrResult:
    micr: str
    fields: dict[str, str | None]
    confidence: float
    per_char: list[float]
    route_to_human: bool
    band_bbox: tuple[int, int, int, int]


def run_pipeline(gray: np.ndarray, recognizer: Recognizer, route_threshold: float = 0.5) -> MicrResult:
    crop, bbox = localize_band(gray)
    text, confs, conf = recognizer(crop)
    return MicrResult(
        micr=text,
        fields=parse_fields(text),
        confidence=conf,
        per_char=confs,
        route_to_human=conf < route_threshold,
        band_bbox=bbox,
    )


class OnnxRecognizer:
    """onnxruntime CRNN backend for the served container (no torch)."""

    def __init__(self, model_path: str | Path = "models/onnx/crnn.onnx"):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, gray_band: np.ndarray) -> tuple[str, list[float], float]:
        x = preprocess_band(gray_band)
        logits = self.session.run(None, {self.input_name: x})[0][0]  # (T, N_LOGITS)
        return greedy_decode(logits)
