from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from detector import Detection, TrackedObject, detect_objects, load_model, reset_tracker, track_objects
from evidence import EvidenceCapture
from violation_detector import ViolationDetector
from violation_engine import ViolationEngine


@dataclass
class PipelineOutput:
    processed_frame: np.ndarray
    violations: list[dict[str, Any]]


class RealTimeViolationPipeline:
    """Single, real-time pipeline that combines detection, tracking, violations, visualization, and evidence."""

    def __init__(
        self,
        model_path: str | None = None,
        stop_line_ratio: float | None = None,
        evidence_dir: str | None = None,
    ) -> None:
        resolved_stop_line_ratio = stop_line_ratio
        if resolved_stop_line_ratio is None:
            resolved_stop_line_ratio = float(os.environ.get("STOP_LINE_Y_RATIO", "0.62"))

        # Keep heavy objects alive for real-time throughput.
        self.model = load_model(model_path)
        reset_tracker()
        self.violation_engine = ViolationEngine(stop_line_ratio=resolved_stop_line_ratio)
        self.visualizer = ViolationDetector(stop_line_ratio=resolved_stop_line_ratio)
        self.evidence_capture = EvidenceCapture(root_dir=evidence_dir)

        self._frame_index = 0

    def capture_frame(self, cap: cv2.VideoCapture) -> np.ndarray | None:
        ok, frame = cap.read()
        if not ok:
            return None
        return frame

    def run_detection(self, frame: np.ndarray) -> list[Detection]:
        return detect_objects(frame, model=self.model)

    def run_tracking(self, frame: np.ndarray, detections: list[Detection]) -> list[TrackedObject]:
        return track_objects(frame, detections)

    def run_violation_engine(
        self,
        tracks: list[TrackedObject],
        frame: np.ndarray,
        timestamp: float,
    ) -> list[dict[str, Any]]:
        engine_input = [
            {
                "id": track.track_id,
                "class": track.class_name,
                "bbox": list(track.box_xyxy),
                "confidence": track.confidence,
            }
            for track in tracks
        ]
        return self.violation_engine.process_frame(
            engine_input,
            frame,
            timestamp=timestamp,
            frame_index=self._frame_index,
        )

    def run_visualization(
        self,
        frame: np.ndarray,
        tracking_violations: list[dict[str, Any]],
        tracks: list[TrackedObject],
    ) -> np.ndarray:
        return self.visualizer.draw_violations(
            frame,
            tracking_violations,
            tracked_objects=tracks,
        )

    def run_evidence_capture(
        self,
        frame: np.ndarray,
        tracking_violations: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        return self.evidence_capture.capture(frame, tracking_violations)

    def process_frame(self, frame: np.ndarray, timestamp: float | None = None) -> PipelineOutput:
        """Pipeline flow:
        1) Capture frame (external or via run_video_stream)
        2) YOLO detection
        3) DeepSORT tracking
        4) ViolationEngine processing
        5) Visualization
        6) Evidence capture

        Returns:
        - processed_frame
        - violation list
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty")

        ts = float(timestamp) if timestamp is not None else time.time()

        detections = self.run_detection(frame)
        tracks = self.run_tracking(frame, detections)
        tracking_violations = self.run_violation_engine(tracks, frame, ts)
        processed = self.run_visualization(frame, tracking_violations, tracks)
        _ = self.run_evidence_capture(frame, tracking_violations)

        self._frame_index += 1
        return PipelineOutput(processed_frame=processed, violations=tracking_violations)

    def run_video_stream(
        self,
        source: int | str = 0,
        display: bool = False,
    ) -> None:
        """Convenience real-time loop for webcam/video file input."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Unable to open source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        try:
            while True:
                frame = self.capture_frame(cap)
                if frame is None:
                    break

                timestamp = self._frame_index / float(fps)
                output = self.process_frame(frame, timestamp=timestamp)

                if display:
                    cv2.imshow("traffic-pipeline", output.processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example: run webcam pipeline and visualize results.
    pipeline = RealTimeViolationPipeline()
    pipeline.run_video_stream(source=0, display=True)
