# MICR E-13B OCR

Reads the **E-13B magnetic-ink character line** at the bottom of a check (routing, account,
check number) under realistic degradation: handwriting overlap, occlusion, skew, blur, noise,
where stock OCR fails.

A portfolio/credential project. Everything is built and trained here: a synthetic E-13B data
generator (glyphs drawn from ISO 1004 geometry, no licensed font), a CRNN+CTC recognizer trained
from scratch, a YOLO11n band detector, a classical baseline ladder, confidence-based human-review
routing, and an ONNX serving service deployed to Cloudflare Containers.

> **Synthetic benchmark, self-generated, no real checks.** Accuracy numbers measure generalization
> across held-out generator configurations, not real-world check reading. E-13B is a constrained
> 14-glyph alphabet; results make no general-document-OCR transfer claim.

## Status

P0 scaffold. See `PLAN.md` for the full build plan and the design decisions behind it.

## License

AGPL-3.0 (the YOLO11n detector uses AGPL-licensed Ultralytics). The served runtime path does not
depend on Ultralytics; closed/production reuse would swap in a permissive detector.
