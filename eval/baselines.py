"""Honest baseline ladder for the recognizer, from no-ML to stock OCR.

- TemplateMatcher: the real non-ML approach for a fixed-pitch known font. Renders the 14
  reference glyphs and classifies each fixed-pitch cell by normalized cross-correlation.
  Strong on clean bands; degrades honestly once skew/overlap break the pitch grid. This is
  the floor the CRNN must beat to mean anything, not a strawman.
- TesseractBaseline: stock Tesseract. It has no E-13B glyphs, so it essentially cannot read
  the band; reported only as a labeled "general OCR cannot read this font" data point, never
  as the thing being beaten.

Both expose the Recognizer interface (gray band -> (text, per_char_conf, seq_conf)).
"""

from __future__ import annotations

import cv2
import numpy as np

from app.decode import INPUT_H
from app.synth.glyphs import ALPHABET, cell_width_px, glyph_ink


class TemplateMatcher:
    def __init__(self, ink_threshold: float = 0.06):
        self.pitch = cell_width_px(INPUT_H)
        self.ink_threshold = ink_threshold
        self.tokens = list(ALPHABET)
        self.templates = np.stack([self._template(t) for t in self.tokens])  # (14, H*pitch)

    def _template(self, token: str) -> np.ndarray:
        g = glyph_ink(token, INPUT_H)  # (gh, gw) ink in [0,1]
        canvas = np.zeros((INPUT_H, self.pitch), np.float32)
        gh, gw = g.shape
        x0 = max(0, (self.pitch - gw) // 2)
        y0 = max(0, (INPUT_H - gh) // 2)
        canvas[y0 : y0 + gh, x0 : x0 + min(gw, self.pitch - x0)] = g[:, : self.pitch - x0]
        v = canvas.flatten()
        n = np.linalg.norm(v)
        return v / n if n else v

    def __call__(self, gray_band: np.ndarray) -> tuple[str, list[float], float]:
        # Binarize (Otsu) so gray ink on textured paper matches the binary templates; the
        # gray DC floor otherwise dominates the normalized correlation. Then crop to the ink's
        # vertical extent so glyph height ~ band height (== a fixed pitch of cell_width_px).
        _, b = cv2.threshold(gray_band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        rows = np.where(b.sum(axis=1) > 0.04 * 255 * b.shape[1])[0]
        if len(rows) == 0:
            return "", [], 1.0
        crop = b[rows.min() : rows.max() + 1, :]
        band = cv2.resize(crop, (max(INPUT_H, round(crop.shape[1] * INPUT_H / crop.shape[0])), INPUT_H))
        ink = (band > 127).astype(np.float32)
        return self._greedy(ink)

    def _greedy(self, ink: np.ndarray) -> tuple[str, list[float], float]:
        """Decode left to right, locally searching each glyph's position (handles pitch
        jitter), advancing by the fixed pitch from the best-aligned cell. Blank cells (the
        inter-field gaps) carry too little ink and are skipped."""
        w = ink.shape[1]
        pitch = self.pitch
        search = max(2, pitch // 4)
        gap = self.ink_threshold * pitch * INPUT_H
        col = ink.sum(axis=0)
        nz = np.where(col > 0.08 * INPUT_H)[0]
        if len(nz) == 0:
            return "", [], 1.0
        x = max(0, int(nz[0]) - pitch // 2)
        chars: list[str] = []
        confs: list[float] = []
        while x < w - pitch * 0.5:
            best = (-1.0, -1, x)
            for dx in range(-search, search + 1):
                xx = x + dx
                if xx < 0 or xx + pitch > w:
                    continue
                cell = ink[:, xx : xx + pitch]
                if cell.sum() < gap:
                    continue
                n = float(np.linalg.norm(cell))
                if n == 0:
                    continue
                score = self.templates @ (cell.flatten() / n)
                k = int(score.argmax())
                if score[k] > best[0]:
                    best = (float(score[k]), k, xx)
            if best[1] < 0:  # blank inter-field cell
                x += pitch
                continue
            chars.append(self.tokens[best[1]])
            confs.append(best[0])
            x = best[2] + pitch
        return "".join(chars), confs, (float(np.mean(confs)) if confs else 1.0)


class TesseractBaseline:
    def __init__(self):
        import pytesseract  # noqa: F401  (imported to fail fast if missing)

        self._pt = pytesseract

    def __call__(self, gray_band: np.ndarray) -> tuple[str, list[float], float]:
        from PIL import Image

        try:  # stock Tesseract has no E-13B glyphs; an error here just means "could not read"
            txt = self._pt.image_to_string(Image.fromarray(gray_band), config="--psm 7")
        except self._pt.TesseractError:
            txt = ""
        cleaned = "".join(ch for ch in txt if ch.isdigit())
        return cleaned, [0.0] * len(cleaned), 0.0


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    import cv2

    return cv2.resize(img, (max(h, round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)
