import numpy as np

from app.synth import render
from app.synth.glyphs import ALPHABET, NUM_CLASSES, cell_width_px, glyph_ink


def test_alphabet_is_14_classes():
    assert NUM_CLASSES == 14
    assert len(ALPHABET) == len(set(ALPHABET)) == 14
    assert set("0123456789").issubset(ALPHABET)


def test_glyph_ink_shape_and_range():
    g = glyph_ink("0", 48)
    assert g.ndim == 2 and g.shape[0] <= 48
    assert 0.0 <= float(g.min()) and float(g.max()) <= 1.0
    assert g.max() > 0.5  # the zero glyph has ink


def test_render_line_dims_and_centers():
    text = "T123456789T"
    ink, centers, cell_w = render.render_line(text, height_px=48, pad_x=5)
    assert ink.shape[0] == 48
    assert ink.shape[1] == cell_w * len(text) + 10
    assert len(centers) == len(text)
    assert cell_w == cell_width_px(48)


def test_space_renders_blank_cell():
    ink, centers, cell_w = render.render_line("0 0", height_px=48)
    # middle cell is blank -> its column band carries no ink
    mid = ink[:, cell_w : 2 * cell_w]
    assert float(mid.sum()) == 0.0
