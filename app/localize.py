"""Classical MICR-band localizer (serving path, no ML, no AGPL).

A check's ruled lines are also wide and horizontal, so aspect ratio alone is not enough.
The MICR band is distinguished by being a row of *many* separate, character-height blobs,
where a ruling is a single thin stroke. The localizer therefore: searches the bottom
region, merges characters into band candidates with a horizontal close, keeps only
candidates that are tall enough and made of several components (rejecting thin rules), then
refines to the band's full horizontal extent by collecting character-height blobs in that
row. Output is a deskewed crop ready for the recognizer. A learned detector (YOLO11n) is
trained separately for the credential and never imported here. An input that already looks
like a cropped band is returned unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


def _deskew(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) < 20:
        return crop
    angle = cv2.minAreaRect(pts)[2]
    if angle < -45:
        angle += 90
    if abs(angle) > 15:
        return crop
    h, w = crop.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(crop, m, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)


def _band_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _strip_rules(mask: np.ndarray, min_h: int, max_w: int, max_h: int) -> np.ndarray:
    """Erase ruled lines / borders before merging glyphs: components too thin to be a glyph,
    too long to be one (horizontal rules), or too tall to be one (vertical rules/borders).
    Vertical removal matters: a surviving full-height border would let the horizontal close
    bridge the band to it and blow up the band's bounding box."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = mask.copy()
    for i in range(1, n):
        _x, _y, cw, ch, _area = stats[i]
        if ch < min_h * 0.6 or cw > max_w or ch > max_h:
            clean[lab == i] = 0
    return clean


def localize_band(gray: np.ndarray) -> tuple[np.ndarray, BBox]:
    """Find and return (deskewed_band_crop, bbox) from a full check grayscale image."""
    h, w = gray.shape
    if h <= w * 0.28:  # already a band-shaped crop
        return _deskew(gray), (0, 0, w, h)

    y0 = int(h * 0.5)
    roi = gray[y0:]
    min_h = max(10, int(0.025 * h))
    mask = _strip_rules(_band_mask(roi), min_h, max_w=int(w * 0.5), max_h=int(roi.shape[0] * 0.5))

    # Merge the surviving glyphs horizontally into a single band blob (bridges field gaps).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, w // 14), 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: BBox | None = None
    best_area = -1
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < w * 0.2 or not (min_h <= ch <= 0.22 * h) or cw / ch < 3.0:
            continue
        n_comp = cv2.connectedComponentsWithStats(mask[y : y + ch, x : x + cw])[0] - 1
        if n_comp < 6:  # a ruled line is one component; the band is many glyphs
            continue
        if cw * ch > best_area:
            best_area, best = cw * ch, (x, y, cw, ch)

    if best is None:  # fallback: the whole bottom strip
        return _deskew(roi), (0, y0, w, h - y0)

    x, y, cw, ch = best
    # Tighten the vertical extent: grow a dense-row run around the band centre. The band
    # is many glyphs per row (dense); distractor text above is sparse and breaks the run.
    col_ink = (mask[:, x : x + cw] > 0).sum(axis=1)
    dense = col_ink > 0.15 * cw
    cy = min(len(dense) - 1, y + ch // 2)
    if dense[cy]:
        top = cy
        while top > 0 and dense[top - 1]:
            top -= 1
        bot = cy
        while bot < len(dense) - 1 and dense[bot + 1]:
            bot += 1
    else:
        top, bot = y, y + ch
    pad = int((bot - top) * 0.25) + 2
    x0, x1 = max(0, x - pad), min(w, x + cw + pad)
    yy0, yy1 = max(0, top - pad), min(roi.shape[0], bot + pad)
    bbox = (x0, y0 + yy0, x1 - x0, yy1 - yy0)
    return _deskew(gray[y0 + yy0 : y0 + yy1, x0:x1]), bbox
