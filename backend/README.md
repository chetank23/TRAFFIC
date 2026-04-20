# Backend API

FastAPI service for traffic violation analysis from uploaded image/video.

## Endpoints

- `GET /health` -> service health check
- `POST /upload` -> run detection and return normalized violations
- `POST /upload/debug` -> run detection and include raw model detections per sampled frame

Enable side-by-side rule-engine output by adding query param `include_rule_engine=true`:

```bash
POST /upload?include_rule_engine=true
POST /upload/debug?include_rule_engine=true
```

Enable DeepSORT tracking output with persistent IDs by adding `include_tracking=true`:

```bash
POST /upload?include_tracking=true
POST /upload/debug?include_tracking=true
```

Tracking output format:

```json
[
	{
		"id": 17,
		"class": "car",
		"bbox": [120.0, 220.0, 360.0, 420.0],
		"confidence": 0.92
	}
]
```

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
- `MIN_VIOLATION_CONFIDENCE`: final violation confidence threshold after dedupe (default `0.60`)
- `HEURISTIC_MIN_TWO_WHEELER_AREA_RATIO`: minimum normalized two-wheeler box area for rider checks (default `0.003`)
- `HEURISTIC_MIN_VEHICLE_CONFIDENCE`: minimum confidence for two-wheeler/rule checks (default `0.55`)
- `HEURISTIC_MIN_RIDER_CONFIDENCE`: minimum rider confidence for head/helmet checks (default `0.50`)
- `DEBUG_MAX_FRAMES`: max sampled video frames returned by `/upload/debug` (default `12`)
- `DEBUG_MAX_DETECTIONS_PER_FRAME`: max detections returned per sampled frame (default `40`)

## Build a traffic-violation model

The bundled `yolov8n.pt` is a generic COCO detector, not a dedicated traffic-violation model.

1. Define classes you actually need: `helmet`, `no_helmet`, `seatbelt`, `no_seatbelt`, `phone_usage`, `red_light_violation`, `triple_riding`, etc.
2. Collect domain data: city CCTV angles, day/night, rain, occlusions, dense traffic.
3. Label with robust rules: consistent boxes, class naming, and clear edge-case policy.
4. Split data: train/val/test with camera and location separation to avoid leakage.
5. Train YOLO on custom labels (example):

```bash
yolo detect train model=yolov8m.pt data=traffic.yaml imgsz=960 epochs=120 batch=16
```

6. Evaluate per-class precision/recall and confusion matrix, then tune class definitions.
7. Export best weights and point backend to it:

```bash
set YOLO_MODEL_PATH=C:\path\to\best.pt
```

8. Keep rule-based heuristics only as fallback; prefer explicit model classes for legal-critical outcomes.

## Test

```bash
cd backend
python -m pytest -q
```
