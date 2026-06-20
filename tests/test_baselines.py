import numpy as np

from app.synth import render
from eval.baselines import TemplateMatcher
from eval.metrics import levenshtein


def test_template_matcher_reads_clean_band():
    label = "T123456789T0012345678U1001"
    ink, _, _ = render.render_line(label, height_px=48, pad_x=8, pad_y=6)
    band = render.ink_to_image(ink)  # clean, no degradation
    pred, confs, conf = TemplateMatcher()(np.asarray(band))
    # The non-ML floor baseline must read clean fixed-pitch E-13B near-perfectly.
    assert levenshtein(pred, label) / len(label) <= 0.1
    assert conf > 0.5
