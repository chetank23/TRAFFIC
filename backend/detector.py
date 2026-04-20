from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


_MODEL = None


def _default_model_path() -> str:
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "models", "yolov8n.pt")


def load_model(model_path: str | None = None):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install backend requirements before running detection."
        ) from exc

    resolved_model_path = model_path or os.environ.get("YOLO_MODEL_PATH") or _default_model_path()
    _MODEL = YOLO(resolved_model_path)
    return _MODEL


def detect_objects(frame, model=None) -> list[Detection]:
    active_model = model or load_model()
    results = active_model(frame, verbose=False)

    detections: list[Detection] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            class_name = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
            detections.append(
                Detection(
                    class_id=cls,
                    class_name=class_name.lower().strip(),
                    confidence=conf,
                    box_xyxy=(x1, y1, x2, y2),
                )
            )

    return detections
