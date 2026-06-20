"""Compose a fixed-pitch E-13B line from glyph ink.

A line is built on a strict character pitch (real E-13B is 8 chars/inch), every glyph
centered in its pitch cell. Fixed pitch is what lets the segment-then-classify baseline
split cleanly, so it is preserved exactly. Output is ink coverage in [0,1] (1 = full ink),
later inked onto a background by the check/degradation stages.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from . import glyphs


def render_line(
    text: str,
    height_px: int = 64,
    pad_x: int = 0,
    pad_y: int = 0,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """Render `text` (tokens from glyphs.ALPHABET) to an ink-coverage array.

    `jitter` perturbs each glyph's x by up to that fraction of the pitch (scan registration
    error); the pitch grid itself stays fixed, so the segmentation baseline still applies.

    Returns (ink, char_centers_px, cell_w_px):
      ink            float32 [0,1], shape (H, W)
      char_centers   x pixel center of each character's pitch cell
      cell_w_px      the fixed pitch in pixels
    """
    cell_w = glyphs.cell_width_px(height_px)
    width = cell_w * len(text) + 2 * pad_x
    height = height_px + 2 * pad_y
    ink = np.zeros((height, width), np.float32)
    centers: list[float] = []
    for i, ch in enumerate(text):
        cell_x0 = pad_x + i * cell_w
        jx = int(rng.uniform(-jitter, jitter) * cell_w) if (jitter and rng is not None) else 0
        centers.append(cell_x0 + cell_w / 2.0)
        if ch == " ":  # blank pitch cell (inter-field gap); advances but draws nothing
            continue
        g = glyphs.glyph_ink(ch, height_px)
        gh, gw = g.shape
        x0 = max(0, min(width - gw, cell_x0 + (cell_w - gw) // 2 + jx))
        y0 = pad_y + (height_px - gh) // 2
        ink[y0 : y0 + gh, x0 : x0 + gw] = np.maximum(ink[y0 : y0 + gh, x0 : x0 + gw], g)
    return ink, centers, cell_w


def ink_to_image(ink: np.ndarray, ink_value: int = 20, bg_value: int = 255) -> Image.Image:
    """Flatten ink coverage onto a solid background -> 8-bit grayscale PIL image."""
    arr = bg_value * (1.0 - ink) + ink_value * ink
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="L")
