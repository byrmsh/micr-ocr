"""Degradation pipeline: turn a clean MICR band into a realistic, hard-to-read scan.

Operates on 8-bit grayscale (255 = paper, 0 = ink). Every effect is label-invariant
(geometry stays mild and centered, so the transcription never changes), which lets the
same image double as a recognizer training crop. Effects are grouped into difficulty
tiers; the generator freezes a tier config before training so "make it harder" cannot be
tuned mid-experiment (see PLAN.md). The headline hard case the job cares about,
handwriting overlapping printed text, is `handwriting_overlay`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Tier:
    name: str
    rotate_deg: float = 0.0  # max |rotation|
    shear: float = 0.0  # max |shear| fraction
    blur_sigma: float = 0.0  # max gaussian blur sigma
    motion_p: float = 0.0  # prob of motion blur
    noise_sigma: float = 0.0  # gaussian noise std (0-255 scale)
    jpeg_q: tuple[int, int] | None = None  # (min,max) quality; None = no jpeg
    morph_p: float = 0.0  # prob of ink thicken/thin
    smudge_n: tuple[int, int] = (0, 0)  # range of smudge blobs
    stamp_p: float = 0.0
    handwriting_p: float = 0.0  # prob of overlapping handwriting strokes
    handwriting_n: tuple[int, int] = (1, 2)
    occlusion_p: float = 0.0
    lighting_p: float = 0.0
    downscale: float = 1.0  # min relative resolution (1.0 = none)


CLEAN = Tier(name="clean")
MEDIUM = Tier(
    name="medium",
    rotate_deg=2.0,
    shear=0.04,
    blur_sigma=0.8,
    motion_p=0.1,
    noise_sigma=6.0,
    jpeg_q=(55, 90),
    morph_p=0.4,
    smudge_n=(0, 2),
    stamp_p=0.1,
    handwriting_p=0.15,
    handwriting_n=(1, 2),
    lighting_p=0.4,
    downscale=0.75,
)
HARD = Tier(
    name="hard",
    rotate_deg=4.0,
    shear=0.08,
    blur_sigma=1.6,
    motion_p=0.3,
    noise_sigma=14.0,
    jpeg_q=(28, 60),
    morph_p=0.7,
    smudge_n=(1, 4),
    stamp_p=0.35,
    handwriting_p=0.7,
    handwriting_n=(2, 4),
    occlusion_p=0.3,
    lighting_p=0.7,
    downscale=0.5,
)
TIERS = {t.name: t for t in (CLEAN, MEDIUM, HARD)}


def _rotate_shear(img: np.ndarray, rng: np.random.Generator, t: Tier) -> np.ndarray:
    h, w = img.shape
    angle = rng.uniform(-t.rotate_deg, t.rotate_deg) if t.rotate_deg else 0.0
    shear = rng.uniform(-t.shear, t.shear) if t.shear else 0.0
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[0, 1] += shear  # add horizontal shear on top of rotation
    return cv2.warpAffine(img, m, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)


def _blur(img: np.ndarray, rng: np.random.Generator, t: Tier) -> np.ndarray:
    if t.motion_p and rng.random() < t.motion_p:
        k = int(rng.integers(3, 9)) | 1
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        if rng.random() < 0.5:
            kernel = kernel.T
        return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    if t.blur_sigma:
        sigma = float(rng.uniform(0.3, t.blur_sigma))
        return cv2.GaussianBlur(img, (0, 0), sigma)
    return img


def _morph(img: np.ndarray, rng: np.random.Generator, t: Tier) -> np.ndarray:
    if not (t.morph_p and rng.random() < t.morph_p):
        return img
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # ink is dark: erode() spreads the dark (thicker ink), dilate() shrinks it (thin/broken).
    return cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)


def _noise(img: np.ndarray, rng: np.random.Generator, t: Tier) -> np.ndarray:
    if not t.noise_sigma:
        return img
    n = rng.normal(0, t.noise_sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _jpeg(img: np.ndarray, rng: np.random.Generator, t: Tier) -> np.ndarray:
    if t.jpeg_q is None:
        return img
    q = int(rng.integers(t.jpeg_q[0], t.jpeg_q[1] + 1))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else img


def _lighting(img: np.ndarray, rng: np.random.Generator, t: Tier) -> np.ndarray:
    if not (t.lighting_p and rng.random() < t.lighting_p):
        return img
    h, w = img.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    d /= d.max() + 1e-6
    gain = 1.0 - rng.uniform(0.15, 0.45) * d  # darken away from a light center
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _smudges(img: np.ndarray, rng: np.random.Generator, t: Tier) -> np.ndarray:
    n = int(rng.integers(t.smudge_n[0], t.smudge_n[1] + 1)) if t.smudge_n[1] else 0
    h, w = img.shape
    out = img.copy()
    for _ in range(n):
        overlay = out.copy()
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        ax, ay = int(rng.uniform(w * 0.04, w * 0.12)), int(rng.uniform(h * 0.1, h * 0.5))
        shade = int(rng.uniform(60, 160))
        cv2.ellipse(overlay, (cx, cy), (ax, ay), rng.uniform(0, 180), 0, 360, shade, -1)
        alpha = float(rng.uniform(0.15, 0.5))
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    return out


def _stamp(img: np.ndarray, rng: np.random.Generator, t: Tier) -> np.ndarray:
    if not (t.stamp_p and rng.random() < t.stamp_p):
        return img
    h, w = img.shape
    overlay = img.copy()
    x0, y0 = int(rng.uniform(0, w * 0.7)), int(rng.uniform(0, h * 0.5))
    x1, y1 = x0 + int(rng.uniform(w * 0.15, w * 0.3)), y0 + int(rng.uniform(h * 0.3, h * 0.8))
    cv2.rectangle(overlay, (x0, y0), (x1, y1), int(rng.uniform(40, 120)), max(1, h // 30))
    alpha = float(rng.uniform(0.2, 0.5))
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def _bezier(p0, p1, p2, p3, steps: int) -> np.ndarray:
    ts = np.linspace(0, 1, steps)[:, None]
    pts = (
        (1 - ts) ** 3 * p0
        + 3 * (1 - ts) ** 2 * ts * p1
        + 3 * (1 - ts) * ts**2 * p2
        + ts**3 * p3
    )
    return pts.astype(np.int32)


def _handwriting(img: np.ndarray, rng: np.random.Generator, t: Tier) -> np.ndarray:
    """Overlay cursive-like ink strokes crossing the band (the job's headline hard case)."""
    if not (t.handwriting_p and rng.random() < t.handwriting_p):
        return img
    h, w = img.shape
    n = int(rng.integers(t.handwriting_n[0], t.handwriting_n[1] + 1))
    overlay = img.copy()
    for _ in range(n):
        x = rng.uniform(0, w * 0.5)
        span = rng.uniform(w * 0.3, w * 0.9)
        pts = []
        for _ in range(rng.integers(2, 4)):  # chain a few Bezier segments
            p0 = np.array([x, rng.uniform(0, h)])
            p1 = np.array([x + span * 0.3, rng.uniform(-h * 0.2, h * 1.2)])
            p2 = np.array([x + span * 0.6, rng.uniform(-h * 0.2, h * 1.2)])
            p3 = np.array([x + span, rng.uniform(0, h)])
            pts.append(_bezier(p0, p1, p2, p3, 40))
            x += span
        curve = np.concatenate(pts)
        shade = int(rng.uniform(20, 90))
        thick = max(1, int(rng.uniform(h * 0.03, h * 0.09)))
        cv2.polylines(overlay, [curve], False, shade, thick, cv2.LINE_AA)
    alpha = float(rng.uniform(0.55, 0.9))  # ink is opaque-ish but lets some print show
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def _occlusion(img: np.ndarray, rng: np.random.Generator, t: Tier) -> np.ndarray:
    if not (t.occlusion_p and rng.random() < t.occlusion_p):
        return img
    h, w = img.shape
    out = img.copy()
    x0 = int(rng.uniform(0, w * 0.8))
    bw = int(rng.uniform(w * 0.05, w * 0.2))
    if rng.random() < 0.5:  # white-out / torn gap
        out[:, x0 : x0 + bw] = 255
    else:  # dark fold line across the band
        y = int(rng.uniform(0, h))
        cv2.line(out, (0, y), (w, y), int(rng.uniform(30, 100)), max(1, h // 20))
    return out


def _downscale(img: np.ndarray, rng: np.random.Generator, t: Tier) -> np.ndarray:
    if t.downscale >= 1.0:
        return img
    f = float(rng.uniform(t.downscale, 1.0))
    h, w = img.shape
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def elastic_warp(img: np.ndarray, rng: np.random.Generator, strength: float = 6.0) -> np.ndarray:
    """Smooth elastic distortion. Held-out-only, so the model is tested on unseen warps."""
    h, w = img.shape
    dx = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), 12) * strength
    dy = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), 12) * strength
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(
        img, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )


def degrade(img: np.ndarray, tier: Tier, rng: np.random.Generator) -> np.ndarray:
    """Apply the tier's degradation chain to a grayscale band. Label is unchanged."""
    img = _handwriting(img, rng, tier)  # ink laid first, then the page is scanned
    img = _smudges(img, rng, tier)
    img = _stamp(img, rng, tier)
    img = _morph(img, rng, tier)
    img = _rotate_shear(img, rng, tier)
    img = _occlusion(img, rng, tier)
    img = _lighting(img, rng, tier)
    img = _blur(img, rng, tier)
    img = _downscale(img, rng, tier)
    img = _noise(img, rng, tier)
    img = _jpeg(img, rng, tier)
    return img
