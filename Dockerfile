FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# opencv-python-headless still needs libglib at runtime; nothing else GUI-related.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps first for layer caching (--no-dev excludes dev/train/baselines
# groups, so torch/ultralytics never enter the serving image).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app
# Only the recognizer ONNX + the calibration json (temperature + serving threshold) are
# needed at runtime; the YOLO ONNX is a training-time artifact and stays out of the image.
COPY models/onnx/crnn.onnx ./models/onnx/crnn.onnx
COPY models/calibration.json ./models/calibration.json
COPY samples ./samples
RUN uv sync --frozen --no-dev

EXPOSE 8080
CMD ["uv", "run", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
