# Backend API

FastAPI service for traffic violation analysis from uploaded image/video.

## Endpoints

- `GET /health` -> service health check
- `POST /upload` -> run detection and return normalized violations

## Setup

```bash
cd backend
python -m pip install -r requirements.txt
```

## Run

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Environment variables

- `YOLO_MODEL_PATH`: path to model file (default `../models/yolov8n.pt`)
- `ALLOWED_ORIGINS`: comma-separated CORS origins

## Test

```bash
cd backend
python -m pytest -q
```
