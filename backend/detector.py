from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


_MODEL = None
_TRACKER = None


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


def _load_tracker():
    global _TRACKER
    if _TRACKER is not None:
        return _TRACKER

    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except Exception as exc:
        raise RuntimeError(
            "deep-sort-realtime is not installed. Install backend requirements before enabling tracking."
        ) from exc

    # max_age handles brief occlusions / temporary misses gracefully.
    _TRACKER = DeepSort(max_age=30, n_init=2)
    return _TRACKER


def reset_tracker() -> None:
    global _TRACKER
    _TRACKER = None


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


def _yolo_to_deepsort_detections(detections: list[Detection]) -> list[list[object]]:
    # Requested intermediate format: [[x1, y1, x2, y2], confidence, class]
    return [
        [[d.box_xyxy[0], d.box_xyxy[1], d.box_xyxy[2], d.box_xyxy[3]], float(d.confidence), d.class_name]
        for d in detections
    ]


def _xyxy_to_ltwh(box_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def track_objects(frame, detections: list[Detection], tracker=None) -> list[TrackedObject]:
    active_tracker = tracker or _load_tracker()

    deepsort_input_xyxy = _yolo_to_deepsort_detections(detections)
    deepsort_input_ltwh = [
        (_xyxy_to_ltwh(box_xyxy), confidence, class_name)
        for box_xyxy, confidence, class_name in deepsort_input_xyxy
    ]

    tracks = active_tracker.update_tracks(deepsort_input_ltwh, frame=frame)

    tracked: list[TrackedObject] = []
    for track in tracks:
        if hasattr(track, "is_confirmed") and not track.is_confirmed():
            continue

        if hasattr(track, "to_ltrb"):
            x1, y1, x2, y2 = [float(v) for v in track.to_ltrb()]
        else:
            continue

        track_class = "unknown"
        if hasattr(track, "get_det_class"):
            cls = track.get_det_class()
            if cls is not None:
                track_class = str(cls)

        track_conf = 0.0
        if hasattr(track, "det_conf") and track.det_conf is not None:
            track_conf = float(track.det_conf)

        tracked.append(
            TrackedObject(
                track_id=int(track.track_id),
                class_name=track_class,
                confidence=track_conf,
                box_xyxy=(x1, y1, x2, y2),
            )
        )

    return tracked
