from __future__ import annotations

import os
from pathlib import Path

import cv2

from detector import detect_objects
from schemas import AnalysisResponse, AnalysisSummary
from violations import FrameContext, detect_violations


VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
STOP_LINE_Y_RATIO = float(os.environ.get("STOP_LINE_Y_RATIO", "0.62"))
STOP_LINE_BAND_RATIO = float(os.environ.get("STOP_LINE_BAND_RATIO", "0.06"))


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _build_summary(total_confidence: float, count: int, unique_types: int) -> AnalysisSummary:
    avg = (total_confidence / count) if count else 0
    return AnalysisSummary(
        total_violations=count,
        unique_types=unique_types,
        avg_confidence=max(0.0, min(1.0, avg)),
    )


def process_image(path: str, file_name: str) -> AnalysisResponse:
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Unable to read image file")

    h, w = image.shape[:2]
    detections = detect_objects(image)
    violations = detect_violations(
        detections,
        FrameContext(
            frame_width=w,
            frame_height=h,
            timestamp=None,
            stop_line_y_ratio=STOP_LINE_Y_RATIO,
            stop_line_band_ratio=STOP_LINE_BAND_RATIO,
        ),
    )

    summary = _build_summary(
        total_confidence=sum(v.confidence for v in violations),
        count=len(violations),
        unique_types=len(set(v.type for v in violations)),
    )

    return AnalysisResponse(
        file_name=file_name,
        is_video=False,
        duration_seconds=None,
        violations=violations,
        summary=summary,
    )


def process_video(path: str, file_name: str) -> AnalysisResponse:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else None

    frame_index = 0
    stride = max(1, int(round(fps / 5.0)))  # sample around 5 frames per second

    violations = []
    previous_vehicle_centers: dict[str, tuple[float, float]] = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_index % stride != 0:
            frame_index += 1
            continue

        h, w = frame.shape[:2]
        detections = detect_objects(frame)

        ctx = FrameContext(
            frame_width=w,
            frame_height=h,
            timestamp=frame_index / fps,
            previous_vehicle_centers=previous_vehicle_centers,
            stop_line_y_ratio=STOP_LINE_Y_RATIO,
            stop_line_band_ratio=STOP_LINE_BAND_RATIO,
        )

        frame_violations = detect_violations(detections, ctx)
        violations.extend(frame_violations)

        # Update center map for speed heuristic.
        vehicles = [d for d in detections if d.class_name in {"car", "truck", "bus", "motorcycle", "bicycle"}]
        previous_vehicle_centers = {
            f"{v.class_name}-{i}": ((v.box_xyxy[0] + v.box_xyxy[2]) / 2.0, (v.box_xyxy[1] + v.box_xyxy[3]) / 2.0)
            for i, v in enumerate(vehicles)
        }

        frame_index += 1

    cap.release()

    # Keep strongest violations and sort by timestamp.
    violations.sort(key=lambda v: (v.timestamp if v.timestamp is not None else 1e9, -v.confidence))

    # Only keep high-confidence violations.
    violations = [v for v in violations if v.confidence >= 0.80]

    # Optional cap to keep payload reasonable.
    violations = violations[:60]

    summary = _build_summary(
        total_confidence=sum(v.confidence for v in violations),
        count=len(violations),
        unique_types=len(set(v.type for v in violations)),
    )

    return AnalysisResponse(
        file_name=file_name,
        is_video=True,
        duration_seconds=duration,
        violations=violations,
        summary=summary,
    )
