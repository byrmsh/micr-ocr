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

    def _decode_phase(self, ink: np.ndarray, phase: int):
        """Tile fixed-pitch cells from `phase`; classify non-empty cells by correlation."""
        w = ink.shape[1]
        chars: list[str] = []
        confs: list[float] = []
        total = 0.0
        x = phase
        gap = self.ink_threshold * self.pitch * INPUT_H
        while x + self.pitch <= w:
            cell = ink[:, x : x + self.pitch]
            if cell.sum() > gap:  # skip blank inter-field cells
                v = cell.flatten()
                n = np.linalg.norm(v)
                if n:
                    score = self.templates @ (v / n)
                    k = int(score.argmax())
                    chars.append(self.tokens[k])
                    confs.append(float(score[k]))
                    total += float(score[k])
            x += self.pitch
        return chars, confs, total

    def __call__(self, gray_band: np.ndarray) -> tuple[str, list[float], float]:
        # Crop to the ink's vertical extent first so glyph height ~ band height; only then is
        # the fixed pitch == cell_width_px(INPUT_H). Skipping this makes padding miscompute it.
        rows = np.where((255 - gray_band.astype(np.float32)).sum(axis=1) > 0.05 * 255 * gray_band.shape[1])[0]
        cropped = gray_band[rows.min() : rows.max() + 1, :] if len(rows) else gray_band
        ink = 1.0 - _resize_h(cropped, INPUT_H).astype(np.float32) / 255.0
        if ink.sum() == 0:
            return "", [], 1.0
        # Brute-force the grid phase: glyphs are centered in their pitch cells, so the right
        # phase puts cell boundaries on the inter-glyph gaps. Pick the phase scoring highest.
        best = max(
            (self._decode_phase(ink, p) for p in range(0, self.pitch, 2)),
            key=lambda r: r[2],
        )
        chars, confs, _ = best
        return "".join(chars), confs, (float(np.mean(confs)) if confs else 1.0)


class TesseractBaseline:
    def __init__(self):
        import pytesseract  # noqa: F401  (imported to fail fast if missing)

        self._pt = pytesseract

    def __call__(self, gray_band: np.ndarray) -> tuple[str, list[float], float]:
        from PIL import Image

        txt = self._pt.image_to_string(
            Image.fromarray(gray_band), config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )
        cleaned = "".join(ch for ch in txt if ch.isdigit())
        return cleaned, [0.0] * len(cleaned), 0.0


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    import cv2

    return cv2.resize(img, (max(h, round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)
