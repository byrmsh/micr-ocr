"""Torch-free CTC decode + recognizer constants.

Kept separate from crnn.py (which imports torch) so the serving runtime can decode ONNX
logits with only numpy. Both the training model and the served model share these.
"""

from __future__ import annotations

import numpy as np

from app.synth.glyphs import IDX_TO_CHAR, NUM_CLASSES

INPUT_H = 32  # recognizer input height; width is variable
BLANK = NUM_CLASSES  # CTC blank index
N_LOGITS = NUM_CLASSES + 1


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def greedy_decode(logits: np.ndarray) -> tuple[str, list[float], float]:
    """CTC greedy decode of one sequence's logits (T, N_LOGITS).

    Returns (text, per_char_confidences, sequence_confidence). Per-char confidence is the
    softmax prob at the timestep that emitted the character; sequence confidence is the mean
    over emitted characters (1.0 for an empty read). Known CTC caveat: the blank-dominated,
    peaky softmax makes these good for *ranking* uncertainty, not as calibrated
    probabilities, until temperature-scaled (see eval/calibration).
    """
    probs = softmax(logits)
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
