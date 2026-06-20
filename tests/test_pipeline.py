"""Routing-layer guards in run_pipeline / serving_params (regressions for the review findings).

The recognizer returns the decoder's seq_conf, which is 1.0 for an empty read; the routing
layer must not let that auto-accept. serving_params must never pair temperature-scaled
confidence with a raw threshold.
"""

import json

import numpy as np

from app.recognizer import run_pipeline, serving_params


def _check_img() -> np.ndarray:
    # Check-shaped grayscale so localize_band returns a real (non-degenerate) crop.
    return np.full((400, 1000), 255, np.uint8)


def test_empty_read_routes_to_human():
    res = run_pipeline(_check_img(), lambda crop: ("", [], 1.0), route_threshold=0.92)
    assert res.micr == ""
    assert res.confidence == 0.0  # not the decoder's 1.0 sentinel for an empty read
    assert res.route_to_human is True


def test_confident_read_auto_accepts():
    res = run_pipeline(_check_img(), lambda crop: ("T123T", [0.99] * 5, 0.99), route_threshold=0.92)
    assert res.route_to_human is False


def test_serving_params_clamps_nonpositive_temperature(tmp_path):
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"temperature": 0, "serving_threshold": 0.9}))
    temperature, threshold = serving_params(p)
    assert temperature == 1.0  # 0 would divide to nan / flip the argmax
    assert threshold == 0.9


def test_serving_params_falls_back_to_legacy_threshold(tmp_path):
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"temperature": 2.4, "threshold": 0.96}))  # no serving_threshold key
    temperature, threshold = serving_params(p)
    assert temperature == 2.4
    assert threshold == 0.96  # legacy strict key, not the raw 0.5 default


def test_serving_params_uncalibrated_defaults(tmp_path):
    temperature, threshold = serving_params(tmp_path / "missing.json")
    assert (temperature, threshold) == (1.0, 0.5)
