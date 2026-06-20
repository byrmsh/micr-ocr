"""Build the demo's bundled samples + pre-computed results (samples/ + index.json).

The served page loads these so the first click returns instantly while the container wakes.
Mixes synthetic checks/bands (clean..hard) with a couple of real public-domain checks, so
the demo honestly shows both the synthetic benchmark and the synthetic-to-real gap. Results
are whatever the real pipeline produces; nothing is faked. Run after the ONNX export:
python -m eval.make_demo_samples
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.recognizer import OnnxRecognizer, run_pipeline, serving_params
from app.synth import check, dataset
from app.synth.config import TRAIN_CONFIG

_REPO = Path(__file__).resolve().parents[1]
SAMPLES = _REPO / "samples"
REAL = _REPO / "assets" / "real_samples"


def _save(name: str, gray: np.ndarray) -> str:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SAMPLES / name), gray)
    return name


def main() -> None:
    temperature, thr = serving_params()
    rec = OnnxRecognizer(_REPO / "models" / "onnx" / "crnn.onnx", temperature=temperature)
    entries: list[dict] = []

    synth = [
        ("Synthetic check", _save("check_normal.png", check.compose_check(101, held_out=False)[0])),
        ("Synthetic check (hard)", _save("check_hard.png", check.compose_check(207, held_out=True)[0])),
        ("Synthetic band (clean)", _save("band_clean.png", dataset.make_band(dataset._seed("demo", 1), "clean", TRAIN_CONFIG)[0])),
        ("Synthetic band (hard)", _save("band_hard.png", dataset.make_band(dataset._seed("demo", 5), "hard", TRAIN_CONFIG)[0])),
    ]
    for label, name in synth:
        gray = cv2.imread(str(SAMPLES / name), cv2.IMREAD_GRAYSCALE)
        res = run_pipeline(gray, rec, thr)
        entries.append(_entry(label, name, res))

    for real in sorted(REAL.glob("*.jpg")) + sorted(REAL.glob("*.png")):
        gray = cv2.imread(str(real), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if gray.shape[1] > 1400:  # downscale large scans
            s = 1400 / gray.shape[1]
            gray = cv2.resize(gray, (1400, int(gray.shape[0] * s)))
        name = _save(f"real_{real.stem}.png", gray)
        res = run_pipeline(gray, rec, thr)
        entries.append(_entry(f"Real check: {real.stem}", name, res))

    (SAMPLES / "index.json").write_text(json.dumps(entries, indent=2))
    print(f"wrote {len(entries)} demo samples to {SAMPLES}")
    for e in entries:
        print(f"  {e['label']:28} -> '{e['micr']}' conf={e['confidence']:.2f} human={e['route_to_human']}")


def _entry(label: str, image: str, res) -> dict:
    return {
        "label": label,
        "image": image,
        "micr": res.micr,
        "fields": res.fields,
        "confidence": round(res.confidence, 4),
        "route_to_human": res.route_to_human,
        "band_bbox": list(res.band_bbox),
    }


if __name__ == "__main__":
    main()
