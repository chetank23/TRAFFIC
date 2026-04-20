from __future__ import annotations

import os
from pathlib import Path

import cv2

from detector import Detection, TrackedObject as DetectorTrackedObject, detect_objects, reset_tracker, track_objects
from evidence import EvidenceCapture
from schemas import (
    AnalysisResponse,
    AnalysisSummary,
    BoundingBox,
    DebugAnalysisResponse,
    EvidenceMetadata,
    FrameDetections,
    RawDetection,
    RuleEngineViolation,
    TrackingViolation,
    TrackedObject,
)
from violation_engine import ViolationEngine
from violation_detector import ViolationDetector
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


def _to_rule_engine_violations(
    rule_violations: list[dict[str, object]], timestamp: float | None = None
) -> list[RuleEngineViolation]:
    converted: list[RuleEngineViolation] = []
    for item in rule_violations:
        bbox_raw = item.get("bbox")
        if not isinstance(bbox_raw, list | tuple) or len(bbox_raw) != 4:
            continue

        violation_timestamp = item.get("timestamp", timestamp)
        if violation_timestamp is not None:
            violation_timestamp = float(violation_timestamp)

        converted.append(
            RuleEngineViolation(
                type=str(item.get("type", "unknown")),
                bbox=(
                    float(bbox_raw[0]),
                    float(bbox_raw[1]),
                    float(bbox_raw[2]),
                    float(bbox_raw[3]),
                ),
                confidence=float(item.get("confidence", 0.0)),
                timestamp=violation_timestamp,
            )
        )

    return converted


def _to_tracked_objects(tracks: list[DetectorTrackedObject]) -> list[TrackedObject]:
    return [
        TrackedObject(
            id=t.track_id,
            class_name=t.class_name,
            bbox=(
                float(t.box_xyxy[0]),
                float(t.box_xyxy[1]),
                float(t.box_xyxy[2]),
                float(t.box_xyxy[3]),
            ),
            confidence=float(t.confidence),
        )
        for t in tracks
    ]


def _to_tracking_violations(items: list[dict]) -> list[TrackingViolation]:
    converted: list[TrackingViolation] = []
    for item in items:
        bbox_raw = item.get("bbox")
        if not isinstance(bbox_raw, list | tuple) or len(bbox_raw) != 4:
            continue

        converted.append(
            TrackingViolation(
                track_id=int(item.get("track_id", -1)),
                type=str(item.get("type", "unknown")),
                bbox=(
                    float(bbox_raw[0]),
                    float(bbox_raw[1]),
                    float(bbox_raw[2]),
                    float(bbox_raw[3]),
                ),
                timestamp=float(item.get("timestamp", 0.0)),
            )
        )

    return converted


def process_image(
    path: str,
    file_name: str,
    include_rule_engine: bool = False,
    include_tracking: bool = False,
    include_violation_engine: bool = False,
) -> AnalysisResponse:
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Unable to read image file")

    h, w = image.shape[:2]
    detections = detect_objects(image)
    rule_engine_violations: list[RuleEngineViolation] | None = None
    tracked_objects: list[TrackedObject] | None = None
    tracking_violations: list[TrackingViolation] | None = None
    evidence_metadata: list[EvidenceMetadata] | None = None

    if include_tracking:
        reset_tracker()
        tracked_objects = _to_tracked_objects(track_objects(image, detections))

    if include_violation_engine:
        if tracked_objects is None:
            reset_tracker()
            tracked_objects = _to_tracked_objects(track_objects(image, detections))

        v_engine = ViolationEngine(stop_line_ratio=STOP_LINE_Y_RATIO)
        engine_input = [
            {
                "id": t.id,
                "class": t.class_name,
                "bbox": list(t.bbox),
                "confidence": t.confidence,
            }
            for t in tracked_objects
        ]
        frame_tracking_violations = v_engine.process_frame(engine_input, image, timestamp=0.0)
        tracking_violations = _to_tracking_violations(frame_tracking_violations)

        evidence = EvidenceCapture()
        raw_metadata = evidence.capture(image, frame_tracking_violations)
        if raw_metadata:
            evidence_metadata = [EvidenceMetadata(**item) for item in raw_metadata]

    if include_rule_engine:
        rule_detector = ViolationDetector(stop_line_ratio=STOP_LINE_Y_RATIO)
        rule_engine_violations = _to_rule_engine_violations(rule_detector.detect_violations(detections, image))

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
        rule_engine_violations=rule_engine_violations,
        tracked_objects=tracked_objects,
        tracking_violations=tracking_violations,
        evidence_metadata=evidence_metadata,
        summary=summary,
    )


def process_image_debug(
    path: str,
    file_name: str,
    include_rule_engine: bool = False,
    include_tracking: bool = False,
    include_violation_engine: bool = False,
) -> DebugAnalysisResponse:
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Unable to read image file")

    h, w = image.shape[:2]
    detections = detect_objects(image)
    rule_engine_violations: list[RuleEngineViolation] | None = None
    tracked_objects: list[TrackedObject] | None = None
    tracking_violations: list[TrackingViolation] | None = None
    evidence_metadata: list[EvidenceMetadata] | None = None

    if include_tracking:
        reset_tracker()
        tracked_objects = _to_tracked_objects(track_objects(image, detections))

    if include_violation_engine:
        if tracked_objects is None:
            reset_tracker()
            tracked_objects = _to_tracked_objects(track_objects(image, detections))

        v_engine = ViolationEngine(stop_line_ratio=STOP_LINE_Y_RATIO)
        engine_input = [
            {
                "id": t.id,
                "class": t.class_name,
                "bbox": list(t.bbox),
                "confidence": t.confidence,
            }
            for t in tracked_objects
        ]
        frame_tracking_violations = v_engine.process_frame(engine_input, image, timestamp=0.0)
        tracking_violations = _to_tracking_violations(frame_tracking_violations)

        evidence = EvidenceCapture()
        raw_metadata = evidence.capture(image, frame_tracking_violations)
        if raw_metadata:
            evidence_metadata = [EvidenceMetadata(**item) for item in raw_metadata]

    if include_rule_engine:
        rule_detector = ViolationDetector(stop_line_ratio=STOP_LINE_Y_RATIO)
        rule_engine_violations = _to_rule_engine_violations(rule_detector.detect_violations(detections, image))

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
        rule_engine_violations=rule_engine_violations,
        tracked_objects=tracked_objects,
        tracking_violations=tracking_violations,
        evidence_metadata=evidence_metadata,
        summary=summary,
        frame_detections=[
            FrameDetections(
                frame_index=0,
                timestamp=0,
                detections=_to_raw_detections(detections, w, h),
                tracked_objects=tracked_objects,
                tracking_violations=tracking_violations,
            )
        ],
    )


def process_video(
    path: str,
    file_name: str,
    include_rule_engine: bool = False,
    include_tracking: bool = False,
    include_violation_engine: bool = False,
) -> AnalysisResponse:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else None

    frame_index = 0
    stride = 1 if (include_tracking or include_violation_engine) else max(1, int(round(fps / 5.0)))

    violations = []
    rule_engine_violations: list[RuleEngineViolation] | None = [] if include_rule_engine else None
    rule_detector = ViolationDetector(stop_line_ratio=STOP_LINE_Y_RATIO) if include_rule_engine else None
    tracked_objects: list[TrackedObject] | None = [] if include_tracking else None
    tracking_violations: list[TrackingViolation] | None = [] if include_violation_engine else None
    violation_engine = ViolationEngine(stop_line_ratio=STOP_LINE_Y_RATIO) if include_violation_engine else None
    evidence_metadata: list[EvidenceMetadata] | None = [] if include_violation_engine else None
    evidence = EvidenceCapture() if include_violation_engine else None

    if include_tracking or include_violation_engine:
        reset_tracker()

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

        frame_tracks: list[TrackedObject] | None = None
        if include_tracking or include_violation_engine:
            frame_tracks = _to_tracked_objects(track_objects(frame, detections))
            if tracked_objects is not None:
                tracked_objects = frame_tracks

        if violation_engine is not None and tracking_violations is not None and frame_tracks is not None:
            engine_input = [
                {
                    "id": t.id,
                    "class": t.class_name,
                    "bbox": list(t.bbox),
                    "confidence": t.confidence,
                }
                for t in frame_tracks
            ]
            frame_tracking_violations = violation_engine.process_frame(
                engine_input,
                frame,
                timestamp=frame_index / fps,
            )
            tracking_violations.extend(_to_tracking_violations(frame_tracking_violations))
            if evidence is not None and evidence_metadata is not None:
                raw_metadata = evidence.capture(frame, frame_tracking_violations)
                evidence_metadata.extend(EvidenceMetadata(**item) for item in raw_metadata)

        if rule_detector is not None and rule_engine_violations is not None:
            frame_rule_violations = rule_detector.detect_violations(detections, frame)
            rule_engine_violations.extend(
                _to_rule_engine_violations(frame_rule_violations, timestamp=frame_index / fps)
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
        rule_engine_violations=rule_engine_violations,
        tracked_objects=tracked_objects,
        tracking_violations=tracking_violations,
        evidence_metadata=evidence_metadata,
        summary=summary,
    )


def process_video_debug(
    path: str,
    file_name: str,
    include_rule_engine: bool = False,
    include_tracking: bool = False,
    include_violation_engine: bool = False,
) -> DebugAnalysisResponse:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else None

    frame_index = 0
    stride = 1 if (include_tracking or include_violation_engine) else max(1, int(round(fps / 5.0)))

    violations = []
    rule_engine_violations: list[RuleEngineViolation] | None = [] if include_rule_engine else None
    rule_detector = ViolationDetector(stop_line_ratio=STOP_LINE_Y_RATIO) if include_rule_engine else None
    tracked_objects: list[TrackedObject] | None = [] if include_tracking else None
    tracking_violations: list[TrackingViolation] | None = [] if include_violation_engine else None
    violation_engine = ViolationEngine(stop_line_ratio=STOP_LINE_Y_RATIO) if include_violation_engine else None
    evidence_metadata: list[EvidenceMetadata] | None = [] if include_violation_engine else None
    evidence = EvidenceCapture() if include_violation_engine else None

    if include_tracking or include_violation_engine:
        reset_tracker()

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

        frame_tracks: list[TrackedObject] | None = None
        frame_tracking_violations: list[TrackingViolation] | None = None
        if include_tracking or include_violation_engine:
            frame_tracks = _to_tracked_objects(track_objects(frame, detections))
            if tracked_objects is not None:
                tracked_objects = frame_tracks

        if violation_engine is not None and frame_tracks is not None:
            engine_input = [
                {
                    "id": t.id,
                    "class": t.class_name,
                    "bbox": list(t.bbox),
                    "confidence": t.confidence,
                }
                for t in frame_tracks
            ]
            frame_tracking_violations = _to_tracking_violations(
                violation_engine.process_frame(
                    engine_input,
                    frame,
                    timestamp=frame_index / fps,
                )
            )
            if tracking_violations is not None:
                tracking_violations.extend(frame_tracking_violations)
            if evidence is not None and evidence_metadata is not None:
                raw_input = [item.model_dump() for item in frame_tracking_violations]
                raw_metadata = evidence.capture(frame, raw_input)
                evidence_metadata.extend(EvidenceMetadata(**item) for item in raw_metadata)

        if rule_detector is not None and rule_engine_violations is not None:
            frame_rule_violations = rule_detector.detect_violations(detections, frame)
            rule_engine_violations.extend(
                _to_rule_engine_violations(frame_rule_violations, timestamp=frame_index / fps)
            )

        if len(frame_detections) < DEBUG_MAX_FRAMES:
            frame_detections.append(
                FrameDetections(
                    frame_index=frame_index,
                    timestamp=frame_index / fps,
                    detections=_to_raw_detections(detections, w, h),
                    tracked_objects=frame_tracks,
                    tracking_violations=frame_tracking_violations,
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
        rule_engine_violations=rule_engine_violations,
        tracked_objects=tracked_objects,
        tracking_violations=tracking_violations,
        evidence_metadata=evidence_metadata,
        summary=summary,
        frame_detections=frame_detections,
    )
