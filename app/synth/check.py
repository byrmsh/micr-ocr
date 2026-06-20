"""Synthetic full-check compositor.

Places a clean E-13B MICR band at the bottom of a check-like canvas amid distractor
elements (bank name, payee/date/amount lines in an ordinary font, a signature scribble),
then applies mild whole-page degradation. Returns the image, the MICR band's axis-aligned
bbox, and the MicrLine. Used to train/evaluate the band detector (YOLO) and to exercise the
classical localizer; it is NOT part of the served runtime.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import degrade as dg
from . import micr, render
from .dataset import _paper


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_distractors(draw: ImageDraw.ImageDraw, w: int, h: int, rng: np.random.Generator) -> None:
    ink = int(rng.integers(40, 90))
    draw.text((int(w * 0.05), int(h * 0.07)), "FIRST NATIONAL BANK", font=_font(int(h * 0.05)), fill=ink)
    draw.text((int(w * 0.70), int(h * 0.07)), "DATE __________", font=_font(int(h * 0.04)), fill=ink)
    draw.text((int(w * 0.05), int(h * 0.34)), "PAY TO THE", font=_font(int(h * 0.035)), fill=ink)
    draw.line((int(w * 0.22), int(h * 0.40), int(w * 0.78), int(h * 0.40)), fill=ink, width=2)
    draw.rectangle((int(w * 0.80), int(h * 0.34), int(w * 0.96), int(h * 0.44)), outline=ink, width=2)
    draw.line((int(w * 0.05), int(h * 0.52), int(w * 0.78), int(h * 0.52)), fill=ink, width=2)
    draw.text((int(w * 0.80), int(h * 0.50)), "DOLLARS", font=_font(int(h * 0.03)), fill=ink)
    draw.line((int(w * 0.60), int(h * 0.70), int(w * 0.92), int(h * 0.70)), fill=ink, width=2)
    draw.text((int(w * 0.62), int(h * 0.71)), "SIGNATURE", font=_font(int(h * 0.028)), fill=ink)


def _rotate_with_bbox(img: np.ndarray, bbox: tuple[int, int, int, int], angle: float):
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(img, m, (w, h), borderValue=235, flags=cv2.INTER_LINEAR)
    x, y, bw, bh = bbox
    corners = np.array([[x, y], [x + bw, y], [x, y + bh], [x + bw, y + bh]], np.float32)
    ones = np.ones((4, 1), np.float32)
    rc = (m @ np.hstack([corners, ones]).T).T
    nx0, ny0 = rc[:, 0].min(), rc[:, 1].min()
    nx1, ny1 = rc[:, 0].max(), rc[:, 1].max()
    return rot, (int(nx0), int(ny0), int(nx1 - nx0), int(ny1 - ny0))


def compose_check(seed: int, held_out: bool = False):
    """Return (check_img uint8, band_bbox (x,y,w,h), MicrLine)."""
    rng = np.random.default_rng(seed)
    w = int(rng.integers(1000, 1400))
    h = int(w / rng.uniform(2.2, 2.6))

    canvas = _paper(h, w, rng, tint_p=0.5 if held_out else 0.3).astype(np.uint8)
    pil = Image.fromarray(canvas, "L")
    draw = ImageDraw.Draw(pil)
    draw.rectangle((4, 4, w - 5, h - 5), outline=int(rng.integers(60, 110)), width=2)
    _draw_distractors(draw, w, h, rng)
    img = np.asarray(pil).copy()

    line = micr.random_micr_line(random.Random(seed))
    band_h = int(rng.integers(28, 44))
    margin = int(w * rng.uniform(0.03, 0.07))
    cov, _, _ = render.render_line(line.render_text, band_h, pad_x=0, pad_y=int(band_h * 0.3))
    bh, bw = cov.shape
    if bw > w - 2 * margin:  # scale the band to fit the check width
        s = (w - 2 * margin) / bw
        cov = cv2.resize(cov, (w - 2 * margin, max(1, int(bh * s))))
        bh, bw = cov.shape
    bx = margin
    by = int(h * rng.uniform(0.85, 0.93)) - bh // 2
    by = max(0, min(h - bh, by))
    ink_value = float(rng.integers(8, 40))
    region = img[by : by + bh, bx : bx + bw].astype(np.float32)
    img[by : by + bh, bx : bx + bw] = (region * (1 - cov) + ink_value * cov).astype(np.uint8)
    bbox = (bx, by, bw, bh)

    tier = dg.HARD if held_out else dg.MEDIUM
    if rng.random() < 0.7:  # mild whole-page skew, bbox tracked
        img, bbox = _rotate_with_bbox(img, bbox, float(rng.uniform(-tier.rotate_deg, tier.rotate_deg)))
    img = dg._lighting(img, rng, tier)
    img = dg._blur(img, rng, tier)
    img = dg._noise(img, rng, tier)
    img = dg._jpeg(img, rng, tier)
    return img, bbox, line
