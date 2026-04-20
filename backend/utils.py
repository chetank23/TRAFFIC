from __future__ import annotations

import os
from pathlib import Path

import cv2

from detector import Detection, detect_objects
from schemas import AnalysisResponse, AnalysisSummary, BoundingBox, DebugAnalysisResponse, FrameDetections, RawDetection
from violations import FrameContext, detect_violations


VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
STOP_LINE_Y_RATIO = float(os.environ.get("STOP_LINE_Y_RATIO", "0.62"))
STOP_LINE_BAND_RATIO = float(os.environ.get("STOP_LINE_BAND_RATIO", "0.06"))
DEBUG_MAX_FRAMES = int(os.environ.get("DEBUG_MAX_FRAMES", "12"))
DEBUG_MAX_DETECTIONS_PER_FRAME = int(os.environ.get("DEBUG_MAX_DETECTIONS_PER_FRAME", "40"))


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


def _normalize_box(
    box: tuple[float, float, float, float], frame_width: int, frame_height: int
) -> BoundingBox:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(frame_width), x1))
    y1 = max(0.0, min(float(frame_height), y1))
    x2 = max(x1 + 1.0, min(float(frame_width), x2))
    y2 = max(y1 + 1.0, min(float(frame_height), y2))

    return BoundingBox(
        x=x1 / frame_width,
        y=y1 / frame_height,
        w=(x2 - x1) / frame_width,
        h=(y2 - y1) / frame_height,
    )


def _to_raw_detections(detections: list[Detection], frame_width: int, frame_height: int) -> list[RawDetection]:
    return [
        RawDetection(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=d.confidence,
            box=_normalize_box(d.box_xyxy, frame_width, frame_height),
        )
        for d in sorted(detections, key=lambda item: item.confidence, reverse=True)[:DEBUG_MAX_DETECTIONS_PER_FRAME]
    ]


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


def process_image_debug(path: str, file_name: str) -> DebugAnalysisResponse:
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

    return DebugAnalysisResponse(
        file_name=file_name,
        is_video=False,
        duration_seconds=None,
        violations=violations,
        summary=summary,
        frame_detections=[
            FrameDetections(
                frame_index=0,
                timestamp=0,
                detections=_to_raw_detections(detections, w, h),
            )
        ],
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


def process_video_debug(path: str, file_name: str) -> DebugAnalysisResponse:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else None

    frame_index = 0
    stride = max(1, int(round(fps / 5.0)))

    violations = []
    previous_vehicle_centers: dict[str, tuple[float, float]] = {}
    frame_detections: list[FrameDetections] = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_index % stride != 0:
            frame_index += 1
            continue

        h, w = frame.shape[:2]
        detections = detect_objects(frame)

        if len(frame_detections) < DEBUG_MAX_FRAMES:
            frame_detections.append(
                FrameDetections(
                    frame_index=frame_index,
                    timestamp=frame_index / fps,
                    detections=_to_raw_detections(detections, w, h),
                )
            )

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

        vehicles = [d for d in detections if d.class_name in {"car", "truck", "bus", "motorcycle", "bicycle"}]
        previous_vehicle_centers = {
            f"{v.class_name}-{i}": ((v.box_xyxy[0] + v.box_xyxy[2]) / 2.0, (v.box_xyxy[1] + v.box_xyxy[3]) / 2.0)
            for i, v in enumerate(vehicles)
        }

        frame_index += 1

    cap.release()

    violations.sort(key=lambda v: (v.timestamp if v.timestamp is not None else 1e9, -v.confidence))
    violations = [v for v in violations if v.confidence >= 0.80]
    violations = violations[:60]

    summary = _build_summary(
        total_confidence=sum(v.confidence for v in violations),
        count=len(violations),
        unique_types=len(set(v.type for v in violations)),
    )

    return DebugAnalysisResponse(
        file_name=file_name,
        is_video=True,
        duration_seconds=duration,
        violations=violations,
        summary=summary,
        frame_detections=frame_detections,
    )
