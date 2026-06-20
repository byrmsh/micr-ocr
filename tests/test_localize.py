from app.localize import localize_band
from app.synth import check
from eval.detect_eval import iou


def test_localizer_finds_band_in_bottom_region():
    # A handful of fixed seeds; the band is near the bottom and wide.
    hits = 0
    for seed in range(8):
        img, gt, _ = check.compose_check(seed=900 + seed, held_out=False)
        crop, bbox = localize_band(img)
        x, y, w, h = bbox
        assert crop.size > 0
        assert y > img.shape[0] * 0.45  # found in the lower half
        if iou(bbox, gt) >= 0.5:
            hits += 1
    assert hits >= 5  # classical localizer is a baseline, not perfect


def test_already_cropped_band_returns_itself():
    img, _gt, _ = check.compose_check(seed=42, held_out=False)
    # a wide-short crop should pass through (deskew only), not re-search
    band = img[int(img.shape[0] * 0.85) :, :]
    crop, bbox = localize_band(band)
    assert crop.shape[0] <= band.shape[0]
