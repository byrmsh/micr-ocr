# MICR-OCR POC — Plan

## Goal

A deployed, clickable, honestly-benchmarked **MICR (E-13B) recognition system** that recovers
the magnetic-ink character line at the bottom of a check under realistic degradation
(handwriting overlap, occlusion, skew, blur, noise) where default OCR fails.

Two deliverables:
1. **Credential artifact** for Computer-Vision / OCR / document-AI Upwork bids. Lets Bayram
   truthfully answer the screening questions ("trained from scratch?", "fine-tuned YOLO?",
   "reproduced a paper?", "OCR systems?", "confidence estimation / human review?").
2. **Deep technical blog post** on bayram.sh referenced from future proposals.

## Honest positioning (what we may claim after the build, and what we may NOT)

CLAIMABLE (because we will actually do it):
- Trained a **CRNN + CTC** sequence recognizer **from scratch** (reproducing the architecture
  from Shi et al. 2015, "An End-to-End Trainable Neural Network for Image-based Sequence
  Recognition"). -> answers "trained from scratch" AND "reproduced a paper".
- **Fine-tuned YOLO (v8n / v11n)** for MICR-band detection. -> answers "fine-tuned YOLO, which version".
- A **synthetic data generation** pipeline (listed as a nice-to-have in the posting).
- **Confidence estimation + human-in-the-loop routing** with calibration (explicitly requested).
- **ONNX** export and a production FastAPI inference service, deployed live.
- Honest accuracy metrics vs a Tesseract baseline on clean / medium / hard splits.

NOT CLAIMABLE (do not assert):
- SOTA results, or accuracy on real customer checks (we use synthetic data, no PII).
- ViT / TrOCR / Donut / Florence-2 trained from scratch (4GB VRAM; at most a small ViT ablation).
- Medical / passport / invoice OCR, or anything we do not build here.

## Constraints (from environment scout, 2026-06-20)

- GPU: RTX 3050 Laptop, **4GB VRAM** -> nano/small models, small batch, mixed precision (AMP),
  gradient checkpointing if needed. No large transformer training.
- RAM ~19GB (only ~4GB free at scout time) -> generate dataset to disk, modest dataloader workers.
- PyTorch via pip CUDA wheels (driver present, no system CUDA toolkit needed). **Pin Python 3.12**
  for torch/ultralytics wheel availability even though the system Python is 3.14.
- Deploy target = **CF Containers** (mirror `pdf-redaction-api`). Serve with **onnxruntime-CPU**,
  NOT torch, to keep the image slim and cold-start acceptable.
- Repo + blog are public -> need a **redistributable E-13B font** (or generate glyphs from the
  ISO 1004 / ANSI X9 geometry). OPEN RISK, see below.

## Architecture / pipeline

1. **Alphabet (14 classes + CTC blank):** digits `0-9` plus the four E-13B symbols
   Transit, Amount, On-Us, Dash. Internal tokens T/A/O/D; render with the E-13B glyphs.
2. **Synthetic generator (`app/synth/`):**
   - Compose valid MICR strings: routing (9 digits wrapped in Transit symbols), account, check
     number (On-Us), amount field (Amount symbols), with correct delimiter placement.
   - Render the band with the E-13B font; place at the check-bottom region.
   - Compose onto a synthetic check canvas: paper texture, background security tint, ruled lines,
     printed labels, a signature scribble, and courtesy/legal handwriting that **overlaps** the
     MICR band.
   - Degradations (tiered clean / medium / hard): affine + perspective skew, Gaussian/motion blur,
     sensor noise, JPEG artifacts, ink bleed/erode (morphology), smudges/stains (alpha blobs),
     stamp overlays, lighting gradient, downscale.
   - Emit per sample: full-check image + MICR-band bbox (YOLO label) + cropped band image + GT
     string (CRNN label) + difficulty tier.
3. **Detection (Stage 1):** YOLOv8n/v11n fine-tune (ultralytics) on full-check images -> MICR-band
   bbox. Crop + box-based deskew -> recognizer input.
4. **Recognition (Stage 2):** CRNN (CNN feature extractor -> map-to-sequence -> BiLSTM -> CTC),
   trained from scratch on band crops. Greedy CTC decode (optional beam).
   - **Ablation baseline:** segment-then-classify CNN (exploits E-13B fixed-pitch monospacing) to
     show the CRNN's robustness advantage under overlap/occlusion. Good experiment + blog narrative.
5. **Confidence + routing:** CTC per-step max-softmax aggregated to per-char and sequence scores;
   temperature calibration; coverage-accuracy curve -> pick a threshold; below it set
   `route_to_human=true`.
6. **Eval harness (`eval/`):** CER, full-line exact-match, per-field accuracy (routing/account/
   check), on clean/medium/hard splits; **Tesseract-default baseline** for the headline
   "recovers X% that default OCR cannot read"; calibration + coverage curves; E-13B confusion matrix.
7. **Serving (`app/main.py`):** FastAPI. `POST /read` (image) -> `{band_bbox, micr_raw,
   parsed:{routing,account,check,amount}, confidence, route_to_human, per_char[]}`; `GET /` demo
   HTML with bundled samples + drag-drop; `GET /health`; `GET /sample/*`. onnxruntime inference.
8. **Deploy:** CF Containers, slim onnxruntime image, `upwork-micr-ocr.bysh.workers.dev`. Mirror
   `pdf-redaction-api` wrangler.jsonc + Dockerfile + `src/index.ts` Worker shim.

## Repo layout (mirrors pdf-redaction-api)

```
micr-ocr/
  app/
    __init__.py
    main.py          # FastAPI serving (onnxruntime)
    pipeline.py      # detect -> crop -> recognize -> confidence -> parse
    synth/           # synthetic data generation
    models/          # CRNN torch def + onnx export helpers
  train/
    train_crnn.py
    train_yolo.py
    export_onnx.py
  eval/
    evaluate.py      # metrics + baseline + curves
  data/              # generated (gitignored); a small sample set committed
  models/onnx/       # exported onnx (committed if small, else GH release)
  src/index.ts       # CF Container worker shim
  tests/
  Dockerfile         # slim onnxruntime serving image
  wrangler.jsonc  pyproject.toml  uv.lock  .python-version(3.12)
  package.json  .npmrc  CLAUDE.md  README.md  .gitignore  .dockerignore
```

## Build phases

- **P0 Scaffold:** repo + uv + deps; FastAPI `/health`; CF deploy skeleton deploys green.
- **P1 Synthetic generator:** clean + tiered degradations; sample gallery.
- **P2 CRNN on clean:** train to near-perfect; stand up the metrics harness.
- **P3 YOLO detector + integration:** train band detector; wire detect->recognize end-to-end.
- **P4 Hard cases:** overlap/occlusion augmentation tuning; confidence calibration + routing;
  Tesseract baseline; honest results table.
- **P5 Serve + deploy:** ONNX export; slim container; live deploy; demo page.
- **P6 Blog post:** E-13B deep dive, methodology, results, honest limitations.

## Risks / open questions (for the flaw-finders and fact-check)

1. **E-13B font licensing** for a public repo + blog (is GnuMICR GPL acceptable? is there an
   OFL/permissive E-13B? or should we generate glyphs from the ISO 1004 geometry to avoid the
   question entirely?). Potential publishing blocker.
2. **CF Containers feasibility:** does the image-size / memory / cold-start envelope comfortably
   fit onnxruntime + opencv + a small model? Confirm limits.
3. **4GB VRAM sufficiency** for YOLOn + CRNN training; confirm realistic batch sizes.
4. **Synthetic-to-real gap:** with no real checks, is "recovers documents default OCR cannot read"
   an honest claim if framed strictly as a synthetic benchmark? Is that credible for the
   credential/blog?
5. **Baseline choice:** is Tesseract-default a fair/honest baseline, or does it strawman? Should we
   add a second baseline (e.g., EasyOCR) so the comparison is not a setup?
6. **Scope creep:** VLM/Florence-2/Donut/SAM are in the posting's nice-to-haves but OUT of scope
   for this POC. Confirm explicit exclusion, or is one of them a cheap, high-value add?
7. **Time/compute realism:** is the P0-P6 arc achievable on this hardware with agent-driven coding
   without an open-ended training-iteration tail?

---

## DECISIONS (final, 2026-06-20, after adversary/alternatives/fact-check review)

Purpose reframed: **portfolio piece**, not a bid for the specific job. Build end to end.

Resolved forks:
- **YOLO: YES.** Train a YOLO11n single-class MICR-band detector as an **isolated training-only**
  deliverable (`train/yolo/`), never imported by the served app. Gives a truthful, fresh
  "fine-tuned YOLO11n" credential. Serving-path detector is **classical OpenCV** localization.
- **Repo license: AGPL-3.0** (ultralytics is AGPL; licensing the public repo AGPL is compliant
  by construction). Blog notes that closed/production reuse would swap in a permissive detector.
- **Real-world anchor: YES.** Download public-domain / sample check images from the internet for a
  **qualitative-only** test (never a headline metric), to show the synthetic-to-real gap honestly.
  If none usable, also print geometry-rendered MICR lines and photograph them.
- **Font: procedural.** Generate the 14 E-13B glyphs from ISO 1004 / ANSI X9.27 grid geometry
  (7x9 matrix, 0.013-inch modules). License-clean. The plan's earlier "Matthew Welch MICR font"
  reference was factually wrong; do not use it. Do not vendor GnuMICR (GPL, no font exception).

Baked-in corrections (no longer optional):
1. **No strawman headline.** Drop any "X% that default OCR cannot read" number. Build a competent
   classical **template / normalized-cross-correlation matcher** as the honest floor. Three-rung
   ladder: template-match (no ML) -> segment-then-classify CNN -> CRNN+CTC. Tesseract-default and
   EasyOCR-default appear only as labeled "stock OCR cannot read E-13B at all" data points;
   optionally add Tesseract WITH public e13b traineddata as a strong reference.
2. **Frozen benchmark.** Freeze the generator before any training; carve out a held-out generator
   config the models never train on; report metrics on THAT split. Label every table + the demo
   page "Synthetic E-13B benchmark (self-generated, no real checks)" in the artifact itself.
3. **Segment-then-classify is co-equal**, not a strawman ablation (use the known fixed pitch so the
   comparison is fair). It is also the MVP-fallback shippable recognizer.
4. **Serving image is onnxruntime-CPU + opencv-headless, NO torch / NO ultralytics.** Separate uv
   dependency groups (serve = main deps; `train` group for torch/torchvision/ultralytics). Measure
   image size / idle RSS / cold-start before choosing CF Containers instance_type (fits basic or
   standard-1: ~0.7-1.5GB, 1-3s cold start). Bundle ONNX weights + samples into the image; add a
   loading spinner + pre-computed sample results so the demo never spins on first click.
5. **MVP cut line at end of P2/P3:** CRNN-from-scratch + classical localization + confidence routing
   + ONNX + live deploy + blog draft is shippable on its own. Hard-case tuning (P4) runs against a
   frozen test set with a target metric band and a hard cap (3 augmentation passes, accept the
   number) to kill the open-ended tail.
6. **Honesty on calibration + scope:** claim the confidence METHOD (temperature scaling +
   coverage-accuracy curve + routing), not a transferable threshold; state the CTC blank-dominated
   max-softmax caveat. State that E-13B is a constrained 14-glyph alphabet; no general-OCR transfer claim.
7. **P0 toolchain gate FIRST:** uv 3.12 venv, torch pinned to a CUDA wheel index, verify
   `torch.cuda.is_available()` before any data/model work.

ONNX export note: export only the logits-producing network (opset 17-19, legacy torch.onnx.export
for the LSTM); keep CTC greedy/beam decode + confidence aggregation in the Python host layer.
```