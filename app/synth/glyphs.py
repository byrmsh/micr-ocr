"""E-13B glyph atlas.

The 14 E-13B glyphs (0-9, Transit, Amount, On-Us, Dash) are vendored as OFL-1.1 SVGs
(assets/fonts/micr-e13b/, by Zachary Schneider, drawn from the ISO 1004 geometry). We
rasterize them once into a cached PNG atlas at a high base scale, then resize on demand
when compositing a line. resvg is needed only to build the atlas; once the PNGs exist
nothing here shells out, so training and serving have no SVG dependency.

Token mapping (one ASCII char per glyph, so a label is a plain string):
    '0'..'9'  digits
    'T'       Transit  (U+2446, brackets the routing number)
    'A'       Amount   (U+2447, brackets the cleared amount)
    'U'       On-Us    (U+2449, separates account / check-number fields)
    '-'       Dash     (U+2448, intra-field separator)
"""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "micr-e13b"
SVG_DIR = _FONT_DIR / "svgs"
PNG_DIR = _FONT_DIR / "png"

TRANSIT, AMOUNT, ONUS, DASH = "T", "A", "U", "-"
SYMBOLS = (TRANSIT, AMOUNT, ONUS, DASH)
ALPHABET = list("0123456789") + list(SYMBOLS)
NUM_CLASSES = len(ALPHABET)  # 14
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}

_SVG_STEM = {str(d): f"u{0x30 + d:04x}" for d in range(10)}
_SVG_STEM.update({TRANSIT: "u2446", AMOUNT: "u2447", DASH: "u2448", ONUS: "u2449"})

# E-13B design grid: full glyph (digit) height and the fixed character advance, in the
# SVG's design units. 1 unit ~ 0.0139 inch; height 8.42u ~ 0.117", pitch 9u ~ 0.125"
# (8 characters per inch), matching the real font.
CELL_H_UNITS = 8.42
PITCH_UNITS = 9.0

# Base atlas resolution: pixels per design unit. High enough that any line height we
# render is a downscale (clean), not an upscale.
BASE_SCALE = 24


def _viewbox(svg_path: Path) -> tuple[float, float]:
    m = re.search(r'viewBox="([-\d.\s]+)"', svg_path.read_text())
    if not m:
        raise ValueError(f"no viewBox in {svg_path}")
    parts = [float(v) for v in m.group(1).split()]
    return parts[2], parts[3]  # width, height


def build_atlas(base_scale: int = BASE_SCALE) -> None:
    """Rasterize each vendored SVG to a grayscale-alpha PNG at base_scale px/unit."""
    resvg = shutil.which("resvg")
    if resvg is None:
        raise RuntimeError(
            "resvg not found; install it to build the glyph atlas "
            "(one-time; the committed PNGs make it unnecessary thereafter)."
        )
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    for token, stem in _SVG_STEM.items():
        svg = SVG_DIR / f"{stem}.svg"
        w_u, h_u = _viewbox(svg)
        w_px, h_px = round(w_u * base_scale), round(h_u * base_scale)
        out = PNG_DIR / f"{stem}.png"
        subprocess.run(
            [resvg, "--width", str(w_px), "--height", str(h_px), str(svg), str(out)],
            check=True,
            capture_output=True,
        )


@lru_cache(maxsize=1)
def _atlas() -> dict[str, np.ndarray]:
    """token -> float32 ink-coverage array in [0,1], shape (h_px, w_px), at BASE_SCALE."""
    if not all((PNG_DIR / f"{stem}.png").exists() for stem in _SVG_STEM.values()):
        build_atlas()
    atlas: dict[str, np.ndarray] = {}
    for token, stem in _SVG_STEM.items():
        img = Image.open(PNG_DIR / f"{stem}.png").convert("RGBA")
        atlas[token] = (np.asarray(img)[:, :, 3].astype(np.float32)) / 255.0
    return atlas


@lru_cache(maxsize=256)
def glyph_ink(token: str, height_px: int) -> np.ndarray:
    """Ink-coverage array in [0,1] for one glyph, scaled so the full cell is height_px tall.

    Short glyphs (Dash, On-Us) keep their true proportions; vertical placement within the
    cell is the caller's job (see render). Shape is (glyph_h_px, glyph_w_px).
    """
    base = _atlas()[token]
    scale = height_px / (CELL_H_UNITS * BASE_SCALE)
    h = max(1, round(base.shape[0] * scale))
    w = max(1, round(base.shape[1] * scale))
    resized = Image.fromarray((base * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
    return np.asarray(resized).astype(np.float32) / 255.0


def cell_width_px(height_px: int) -> int:
    """Fixed advance (pitch) in pixels for a line whose cell height is height_px."""
    return round(PITCH_UNITS / CELL_H_UNITS * height_px)
