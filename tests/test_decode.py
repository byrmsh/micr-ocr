import numpy as np

from app.decode import BLANK, N_LOGITS, greedy_decode, seq_conf_at_temp, softmax
from app.synth.glyphs import CHAR_TO_IDX


def _onehot_logits(indices: list[int], scale: float = 10.0) -> np.ndarray:
    logits = np.zeros((len(indices), N_LOGITS), np.float32)
    for t, idx in enumerate(indices):
        logits[t, idx] = scale
    return logits


def test_greedy_decode_collapses_repeats_and_blanks():
    seq = [CHAR_TO_IDX["1"], CHAR_TO_IDX["1"], BLANK, CHAR_TO_IDX["1"], CHAR_TO_IDX["2"]]
    text, confs, conf = greedy_decode(_onehot_logits(seq))
    assert text == "112"  # repeat collapsed, blank dropped, then a fresh 1 and 2
    assert len(confs) == 3
    assert 0.0 < conf <= 1.0


def test_empty_decode_is_confident_by_convention():
    text, confs, conf = greedy_decode(_onehot_logits([BLANK, BLANK]))
    assert text == "" and confs == [] and conf == 1.0


def test_softmax_normalizes():
    p = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert abs(float(p.sum()) - 1.0) < 1e-6


def test_temperature_lowers_confidence():
    rows = _onehot_logits([CHAR_TO_IDX["7"], CHAR_TO_IDX["8"]])
    assert seq_conf_at_temp(rows, 1.0) > seq_conf_at_temp(rows, 5.0)
