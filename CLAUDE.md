# micr-ocr (portfolio POC)

MICR E-13B check OCR, built end to end as a portfolio/credential piece (not a bid for one
specific job). Reads the magnetic-ink line at the bottom of a check (routing, account, check
number) under handwriting overlap, occlusion, skew, blur, and noise, where stock OCR fails.

Everything is trained here from scratch on self-generated data. The point is to back the
recurring CV/OCR/document-AI screening questions with real, citable work: trained-from-scratch
CRNN, fine-tuned YOLO11n, synthetic data generation, confidence/human-review routing, ONNX
serving. See `PLAN.md` for the full plan and the design decisions (and the adversarial review
that shaped them).

## Honesty constraints (hard; baked into the artifact, not just prose)

- Synthetic benchmark, self-generated, no real checks. Every results table and the demo page
  says so. Accuracy is reported on a **held-out generator config** the models never train on,
  so it measures cross-generator generalization, not real-world check reading.
- E-13B is a constrained 14-glyph alphabet. No general-document-OCR transfer claim.
- Baseline ladder is honest: a competent classical template matcher is the floor; stock
  Tesseract/EasyOCR appear only as labeled "general OCR cannot read E-13B" points, never as the
  thing being beaten.
- Confidence: claim the METHOD (temperature scaling + coverage-accuracy + routing), not a
  transferable threshold. The CTC blank-dominated-softmax caveat is stated.
- The few real check images (`assets/real_samples/`, public-domain / CC) are a qualitative
  anchor only, never a headline metric.

## Stack and licensing

- Python 3.12 (pinned; system Python is newer and lacks torch wheels). `uv` for deps.
- Serving runtime = onnxruntime-CPU + opencv-headless, **no torch, no ultralytics** (kept in
  the non-default `train` dependency group so the deploy image stays slim).
- Repo is **AGPL-3.0**: ultralytics (YOLO11n) is AGPL, and it is trained here. The served
  runtime does not import it; the serving-path detector is the classical OpenCV localizer.
- E-13B glyphs: vendored OFL-1.1 SVGs (`assets/fonts/micr-e13b/`, by Zachary Schneider, drawn
  from ISO 1004 geometry), rasterized once into a committed PNG atlas. No proprietary font.

## Commands (run from repo root; training needs the GPU)

```bash
# data (frozen train/val/test families + a held-out generator the models never train on)
uv run --no-group dev python -m app.synth.dataset --train 30000 --val 3000 --test 3000 --heldout 3000

# recognizer (CRNN+CTC from scratch; crop-jitter aug for detector-crop robustness)
uv run --no-group dev python -m train.train_crnn --epochs 30 --batch 32
uv run --no-group dev python -m train.export_crnn_onnx          # -> models/onnx/crnn.onnx (parity-checked)

# detector (isolated, training-only; ultralytics never enters serving)
uv run --no-group dev python -m train.yolo.make_yolo_data --train 5000 --val 800
uv run --no-group dev python -m train.yolo.train_yolo
uv run --no-group dev python -m train.yolo.export_onnx

# evaluation for the blog
uv run --no-group dev python -m eval.benchmark --split heldout      # recognizer ladder by tier
uv run --no-group dev python -m eval.detect_eval                    # classical vs YOLO IoU
uv run --no-group dev python -m eval.calibration --split heldout    # temperature + routing threshold
uv run --no-group dev python -m eval.make_demo_samples             # bundle demo samples + results

# tests (torch-free; fast)
uv run pytest
```

## Serving / deploy

FastAPI (`app/main.py`): `GET /health`, `POST /read` (multipart image -> recognized MICR,
parsed fields, confidence, route_to_human, band bbox), `GET /` demo page with bundled
pre-computed sample results (so first click is instant while the container wakes),
`GET /sample/*`. Mirrors the `pdf-redaction-api` Cloudflare Containers deploy
(Dockerfile + wrangler.jsonc + `src/index.ts` Durable-Object Container shim).

Deploy: `pnpm install` then `pnpm deploy`. Worker name `upwork-micr-ocr`
(-> `https://upwork-micr-ocr.bysh.workers.dev`). Do not deploy or send to anyone without
asking first. `instance_type` is chosen from a measured image-size / RSS / cold-start check
before deploy, not copied.

## Gotchas

- 4GB VRAM (RTX 3050 laptop): CRNN channels are kept light; YOLO uses `mosaic=0`, small batch.
- The recognizer trains on band crops (`app.synth.dataset`); end-to-end on full checks goes
  through the classical localizer, whose looser crops are the main band-only vs e2e gap. The
  crop-jitter augmentation (random crop between ink bbox and full margin, never cutting glyphs)
  is what closes it; do not remove it.
- The glyph atlas auto-builds from the SVGs via `resvg` on first use; the committed PNGs make
  `resvg` unnecessary thereafter (and absent from training/serving).
- Generation is multiprocess; it competes with the training dataloader for CPU. Generate data
  before training, or cap workers.
