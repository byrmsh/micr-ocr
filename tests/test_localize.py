from app.localize import localize_band
from app.synth import check


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (aw * ah + bw * bh - inter + 1e-6)


def test_localizer_finds_band_in_bottom_region():
    # A handful of fixed seeds; the band is near the bottom and wide.
    hits = 0
    for seed in range(8):
        img, gt, _ = check.compose_check(seed=900 + seed, held_out=False)
        crop, bbox = localize_band(img)
        x, y, w, h = bbox
        assert crop.size > 0
        assert y > img.shape[0] * 0.45  # found in the lower half
        if _iou(bbox, gt) >= 0.5:
            hits += 1
    assert hits >= 5  # classical localizer is a baseline, not perfect


def test_already_cropped_band_returns_itself():
    img, _gt, _ = check.compose_check(seed=42, held_out=False)
    # a wide-short crop should pass through (deskew only), not re-search
    band = img[int(img.shape[0] * 0.85) :, :]
    crop, bbox = localize_band(band)
    assert crop.shape[0] <= band.shape[0]
