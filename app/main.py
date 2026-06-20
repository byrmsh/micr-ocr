"""FastAPI serving entrypoint (onnxruntime; no torch, no ultralytics).

POST /read an image (a full check or a cropped MICR band) and get back the recognized E-13B
line, parsed fields, a confidence, and a route-to-human flag. The demo page bundles
pre-computed sample results so the first click returns instantly even while the container
wakes from idle.
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

from app.recognizer import OnnxRecognizer, run_pipeline

_APP_DIR = Path(__file__).resolve().parent
_REPO = _APP_DIR.parent
_MODEL = _REPO / "models" / "onnx" / "crnn.onnx"
_SAMPLES = _REPO / "samples"
_CALIB = _REPO / "models" / "calibration.json"

app = FastAPI(
    title="MICR E-13B OCR",
    description="Reads the E-13B magnetic-ink line at the bottom of a check. "
    "Synthetic benchmark, self-generated, no real checks.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def _calib() -> dict:
    return json.loads(_CALIB.read_text()) if _CALIB.exists() else {}


@lru_cache(maxsize=1)
def _recognizer() -> OnnxRecognizer:
    if not _MODEL.exists():
        raise HTTPException(503, "model not bundled in this build")
    return OnnxRecognizer(_MODEL, temperature=float(_calib().get("temperature", 1.0)))


@lru_cache(maxsize=1)
def _threshold() -> float:
    return float(_calib().get("serving_threshold", 0.5))


def _load_gray(data: bytes) -> np.ndarray:
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("L"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"could not decode image: {exc}") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": _MODEL.exists()}


@app.post("/read")
async def read(file: UploadFile = File(...)) -> JSONResponse:
    gray = _load_gray(await file.read())
    res = run_pipeline(gray, _recognizer(), _threshold())
    return JSONResponse(
        {
            "micr": res.micr,
            "fields": res.fields,
            "confidence": round(res.confidence, 4),
            "route_to_human": res.route_to_human,
            "band_bbox": list(res.band_bbox),
        }
    )


@app.get("/sample/{name}")
def sample(name: str) -> FileResponse:
    path = (_SAMPLES / name).resolve()
    if path.parent != _SAMPLES.resolve() or not path.exists():
        raise HTTPException(404, "no such sample")
    return FileResponse(path)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MICR E-13B OCR</title>
<style>
 :root{--bg:#0f1115;--card:#1a1d24;--ink:#e8eaed;--mut:#9aa0aa;--ok:#3fb950;--warn:#d29922;--accent:#58a6ff}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}
 .wrap{max-width:860px;margin:0 auto;padding:32px 20px 64px}
 h1{font-size:24px;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 8px}
 .note{font-size:13px;color:var(--mut);background:#161a22;border:1px solid #232833;border-radius:8px;padding:10px 12px;margin:14px 0}
 .drop{border:2px dashed #2c313c;border-radius:12px;padding:26px;text-align:center;cursor:pointer;background:var(--card);transition:.15s}
 .drop:hover{border-color:var(--accent)} .drop input{display:none}
 .samples{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
 .samples button{background:var(--card);color:var(--ink);border:1px solid #2c313c;border-radius:8px;padding:7px 11px;cursor:pointer;font-size:13px}
 .samples button:hover{border-color:var(--accent)}
 canvas{max-width:100%;border-radius:8px;margin-top:14px;display:none;background:#000}
 .res{background:var(--card);border:1px solid #232833;border-radius:12px;padding:16px;margin-top:14px;display:none}
 .micr{font-family:ui-monospace,Menlo,monospace;font-size:20px;letter-spacing:1px;word-break:break-all}
 .row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
 .chip{background:#161a22;border:1px solid #232833;border-radius:7px;padding:6px 10px;font-size:13px}
 .chip b{color:var(--mut);font-weight:500;margin-right:5px}
 .flag{font-weight:600} .flag.ok{color:var(--ok)} .flag.warn{color:var(--warn)}
 a{color:var(--accent)} code{background:#161a22;padding:1px 5px;border-radius:4px}
</style></head><body><div class="wrap">
 <h1>MICR E-13B OCR</h1>
 <p class="sub">Reads the magnetic-ink line at the bottom of a check: routing, account, check number.</p>
 <div class="note"><b>Synthetic benchmark, self-generated, no real checks.</b> A CRNN+CTC recognizer
  trained from scratch on procedurally degraded E-13B, with a classical band localizer and
  confidence-based human-review routing. Upload a check or a cropped MICR band.</div>
 <label class="drop" id="drop">Drop an image here or click to choose<input type="file" id="file" accept="image/*"></label>
 <div class="samples" id="samples"></div>
 <canvas id="cv"></canvas>
 <div class="res" id="res">
   <div class="micr" id="micr"></div>
   <div class="row" id="fields"></div>
   <div class="row"><span class="chip"><b>confidence</b><span id="conf"></span></span>
     <span class="chip"><b>routing</b><span class="flag" id="route"></span></span></div>
 </div>
 <p class="note" style="margin-top:22px">E-13B is a constrained 14-glyph alphabet; results make no
   general-document-OCR claim. <a href="/docs">API docs</a> · <code>POST /read</code> (multipart image).</p>
<script>
const $=s=>document.querySelector(s), cv=$("#cv"), ctx=cv.getContext("2d");
async function loadSamples(){try{const r=await fetch("/sample/index.json");if(!r.ok)return;
 const items=await r.json();const box=$("#samples");
 items.forEach(it=>{const b=document.createElement("button");b.textContent=it.label;
  b.onclick=()=>runSample(it);box.appendChild(b);});}catch(e){}}
function draw(img,bbox){cv.width=img.naturalWidth;cv.height=img.naturalHeight;cv.style.display="block";
 ctx.drawImage(img,0,0);if(bbox){ctx.strokeStyle="#58a6ff";ctx.lineWidth=Math.max(2,img.naturalWidth/300);
  ctx.strokeRect(bbox[0],bbox[1],bbox[2],bbox[3]);}}
function show(res){$("#res").style.display="block";$("#micr").textContent=res.micr||"(nothing read)";
 const f=res.fields||{};$("#fields").innerHTML=Object.entries(f).map(([k,v])=>
  `<span class="chip"><b>${k}</b>${v??"-"}</span>`).join("");
 $("#conf").textContent=(res.confidence*100).toFixed(1)+"%";
 const rt=$("#route");rt.textContent=res.route_to_human?"send to human review":"auto-accept";
 rt.className="flag "+(res.route_to_human?"warn":"ok");}
async function runSample(it){const img=new Image();img.onload=()=>draw(img,it.band_bbox);img.src="/sample/"+it.image;show(it);}
async function upload(file){const img=new Image();img.onload=()=>draw(img,null);img.src=URL.createObjectURL(file);
 const fd=new FormData();fd.append("file",file);const r=await fetch("/read",{method:"POST",body:fd});
 if(!r.ok){alert("read failed: "+r.status);return;}const res=await r.json();
 const i2=new Image();i2.onload=()=>draw(i2,res.band_bbox);i2.src=URL.createObjectURL(file);show(res);}
$("#drop").onclick=()=>$("#file").click();
$("#file").onchange=e=>e.target.files[0]&&upload(e.target.files[0]);
$("#drop").ondragover=e=>{e.preventDefault();};
$("#drop").ondrop=e=>{e.preventDefault();e.dataTransfer.files[0]&&upload(e.dataTransfer.files[0]);};
loadSamples();
</script></div></body></html>"""
