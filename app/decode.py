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


def greedy_decode(logits: np.ndarray, temperature: float = 1.0) -> tuple[str, list[float], float]:
    """CTC greedy decode of one sequence's logits (T, N_LOGITS).

    Returns (text, per_char_confidences, sequence_confidence). Per-char confidence is the
    softmax prob at the timestep that emitted the character; sequence confidence is the mean
    over emitted characters (1.0 for an empty read). Known CTC caveat: the blank-dominated,
    peaky softmax makes these good for *ranking* uncertainty, not as calibrated
    probabilities, until temperature-scaled (see eval/calibration).

    `temperature` divides the logits before softmax. The decoded text is unchanged (argmax is
    temperature-invariant); only the confidences are. Serving passes the calibration-fit
    temperature so the returned confidence lives in the same space as the routing threshold.
    """
    probs = softmax(logits / temperature)
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


def emitted_logits(logits: np.ndarray) -> tuple[str, np.ndarray]:
    """Greedy decode returning the logit rows at the emitted timesteps (T_emit, N_LOGITS).

    Keeps the raw logits so confidence can be recomputed at any temperature during
    calibration without re-running the model.
    """
    best = logits.argmax(axis=1)
    chars: list[str] = []
    rows: list[np.ndarray] = []
    prev = BLANK
    for t, idx in enumerate(best):
        if idx != BLANK and idx != prev:
            chars.append(IDX_TO_CHAR[int(idx)])
            rows.append(logits[t])
        prev = idx
    return "".join(chars), (np.array(rows) if rows else np.zeros((0, logits.shape[1])))


def seq_conf_at_temp(rows: np.ndarray, temperature: float) -> float:
    """Mean per-character max-softmax over emitted logit rows at the given temperature."""
    if len(rows) == 0:
        return 1.0
    return float(softmax(rows / temperature).max(axis=1).mean())
