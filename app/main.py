"""FastAPI serving entrypoint.

P0 scaffold: only /health is live. The recognition pipeline (/read) and the demo
page are wired in P5 once the ONNX models exist. Inference runs on onnxruntime-CPU;
torch and ultralytics are intentionally absent from the serving image.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="MICR E-13B OCR",
    description="Reads the E-13B magnetic-ink line at the bottom of a check.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return "<!doctype html><title>MICR E-13B OCR</title><h1>MICR E-13B OCR</h1><p>Scaffold. Demo lands in P5.</p>"
