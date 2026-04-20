from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TrackState:
    class_name: str
    bbox: tuple[float, float, float, float]
    centers: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=12))
    last_seen_ts: float = 0.0
    missed_frames: int = 0


class ViolationEngine:
    """Modular temporal violation engine built on tracked detections.

    Expected track item shape:
    {
      "id": int,
      "class": str,
      "bbox": [x1, y1, x2, y2],
      "confidence": float
    }
    """

    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
    RED_LABELS = {"red", "red_light", "traffic light red", "red signal"}
    MIN_RIDER_MOTORCYCLE_IOU = 0.2
    MIN_PERSON_CONFIDENCE = 0.35
    MIN_MOTORCYCLE_CONFIDENCE = 0.35
    MIN_PERSON_BOX_HEIGHT = 30.0

    def __init__(
        self,
        stop_line_ratio: float = 0.62,
        max_missed_frames: int = 25,
        violation_cooldown_frames: int = 45,
    ) -> None:
        self.history: dict[int, TrackState] = {}
        self.stop_line_ratio = stop_line_ratio
        self.max_missed_frames = max_missed_frames
        self.violation_cooldown_frames = max(1, violation_cooldown_frames)
        self.traffic_light_state: str | None = None
        # Required dedupe structure: {track_id: [violations already detected]}.
        # Internally stored as sets for efficient O(1) membership checks.
        self.detected_violations: dict[int, set[str]] = {}
        # Frame index of last emission per (track, violation_type).
        self._violation_last_frame: dict[int, dict[str, int]] = {}
        self._seatbelt_classifier: Callable[[object, dict, dict, dict], bool | float] = self._default_seatbelt_classifier

    def set_seatbelt_classifier(self, classifier: Callable[[object, dict, dict, dict], bool | float]) -> None:
        """Inject a custom seatbelt classifier.

        Classifier signature:
        (driver_crop, driver_track, car_track, context) -> bool | float

        - bool: True means seatbelt present
        - float: probability in [0, 1], where >= 0.5 means seatbelt present
        """
        self._seatbelt_classifier = classifier

    def set_traffic_light_state(self, state: str | None) -> None:
        self.traffic_light_state = state.lower().strip() if state else None

    def process_frame(
        self,
        tracks: list[dict],
        frame,
        timestamp: float | None = None,
        frame_index: int | None = None,
    ) -> list[dict]:
        ts = float(timestamp) if timestamp is not None else time.time()
        frame_no = int(frame_index) if frame_index is not None else int(round(ts * 30.0))
        frame_h = frame.shape[0] if frame is not None else 0
        stop_line_y = frame_h * max(0.0, min(1.0, self.stop_line_ratio))

        parsed_tracks = [t for t in (self._parse_track(x) for x in tracks) if t is not None]
        self._update_history(parsed_tracks, ts)

        violations: list[dict] = []
        violations.extend(self._check_helmet_rule(parsed_tracks, ts, frame_no))
        violations.extend(self._check_red_light_rule(parsed_tracks, stop_line_y, ts, frame_no))
        violations.extend(self._check_seatbelt_rule(parsed_tracks, frame, ts, frame_no))

        self._cleanup_history()
        return violations

    def _check_helmet_rule(self, tracks: list[dict], timestamp: float, frame_index: int) -> list[dict]:
        return self.detect_no_helmet_violations(tracks, timestamp, frame_index)

    def detect_no_helmet_violations(
        self,
        tracks: list[dict],
        timestamp: float,
        frame_index: int | None = None,
    ) -> list[dict]:
        current_frame = int(frame_index) if frame_index is not None else int(round(float(timestamp) * 30.0))
        people = [t for t in tracks if t["class"] == "person"]
        helmets = [t for t in tracks if t["class"] == "helmet"]
        motorcycles = [t for t in tracks if t["class"] == "motorcycle"]

        violations: list[dict] = []
        seen_track_ids: set[int] = set()

        for person in people:
            if person["id"] in seen_track_ids:
                continue

            # Skip very weak/partial person detections to reduce false positives.
            if person["confidence"] < self.MIN_PERSON_CONFIDENCE:
                continue
            if (person["bbox"][3] - person["bbox"][1]) < self.MIN_PERSON_BOX_HEIGHT:
                continue

            rider_bike = self._find_associated_vehicle(
                person["bbox"],
                motorcycles,
                min_iou=self.MIN_RIDER_MOTORCYCLE_IOU,
            )
            if rider_bike is None:
                continue

            if rider_bike["confidence"] < self.MIN_MOTORCYCLE_CONFIDENCE:
                continue

            head_box = self._extract_head_region(person["bbox"])
            has_helmet = any(
                self._iou(head_box, helmet["bbox"]) > 0.0
                or self._point_in_box(self._center(helmet["bbox"]), head_box)
                for helmet in helmets
            )

            if has_helmet:
                continue

            if not self._should_emit_violation(person["id"], "no_helmet", current_frame):
                continue

            seen_track_ids.add(person["id"])
            violations.append(
                {
                    "track_id": person["id"],
                    "type": "no_helmet",
                    "bbox": list(person["bbox"]),
                    "timestamp": timestamp,
                }
            )

        return violations

    def _check_red_light_rule(
        self,
        tracks: list[dict],
        stop_line_y: float,
        timestamp: float,
        frame_index: int,
    ) -> list[dict]:
        traffic_state = self.traffic_light_state
        if traffic_state is None:
            traffic_state = "red" if self._is_red_light_active(tracks) else "green"

        raw = self.detect_red_light_violations(
            tracks=tracks,
            traffic_light_state=traffic_state,
            frame_index=frame_index,
            stop_line_y=stop_line_y,
        )

        track_by_id = {t["id"]: t for t in tracks}
        violations: list[dict] = []
        for item in raw:
            track = track_by_id.get(item["track_id"])
            if track is None:
                continue
            violations.append(
                {
                    "track_id": item["track_id"],
                    "type": item["type"],
                    "bbox": list(track["bbox"]),
                    "timestamp": timestamp,
                    "frame": item["frame"],
                }
            )

        return violations

    def detect_red_light_violations(
        self,
        tracks: list[dict],
        traffic_light_state: str,
        frame_index: int,
        stop_line_y: float,
    ) -> list[dict]:
        if traffic_light_state.lower().strip() != "red":
            return []

        violations: list[dict] = []
        for track in tracks:
            track_id = track["id"]
            if track["class"] not in {"car", "motorcycle"}:
                continue

            state = self.history.get(track_id)
            if state is None or len(state.centers) < 2:
                continue

            prev_y = state.centers[-2][1]
            curr_y = state.centers[-1][1]
            crossed_line = prev_y < stop_line_y <= curr_y
            if not crossed_line:
                continue

            if not self._should_emit_violation(track_id, "red_light_violation", frame_index):
                continue

            violations.append(
                {
                    "type": "red_light_violation",
                    "track_id": track_id,
                    "frame": int(frame_index),
                }
            )

        return violations

    def _check_seatbelt_rule(self, tracks: list[dict], frame, timestamp: float, frame_index: int) -> list[dict]:
        return self.detect_no_seatbelt_violations(tracks, frame, timestamp, frame_index)

    def detect_no_seatbelt_violations(
        self,
        tracks: list[dict],
        frame,
        timestamp: float,
        frame_index: int | None = None,
    ) -> list[dict]:
        """Detect no-seatbelt violations using a hybrid rule + classifier hook.

        Steps:
        1) Find driver region in each car (front-left region).
        2) Match a person as driver candidate inside that region.
        3) Crop driver region and run pluggable classifier hook.
        4) If no seatbelt -> emit violation.
        """
        if frame is None:
            return []
        current_frame = int(frame_index) if frame_index is not None else int(round(float(timestamp) * 30.0))

        cars = [t for t in tracks if t["class"] == "car"]
        persons = [t for t in tracks if t["class"] == "person"]
        seatbelts = [t for t in tracks if t["class"] == "seatbelt"]

        violations: list[dict] = []
        seen_driver_ids: set[int] = set()

        for car in cars:
            driver_region = self._extract_driver_region(car["bbox"])
            driver_candidates = [
                p
                for p in persons
                if self._iou(p["bbox"], car["bbox"]) > 0.05
                and (
                    self._iou(p["bbox"], driver_region) > 0.03
                    or self._point_in_box(self._center(p["bbox"]), driver_region)
                )
            ]
            if not driver_candidates:
                continue

            driver = max(driver_candidates, key=lambda p: self._iou(p["bbox"], driver_region))
            if driver["id"] in seen_driver_ids:
                continue

            crop = self._crop_region(frame, driver_region)
            if crop is None:
                continue

            context = {
                "seatbelts": seatbelts,
                "driver_region": driver_region,
            }
            classifier_result = self._seatbelt_classifier(crop, driver, car, context)
            has_seatbelt = bool(classifier_result) if isinstance(classifier_result, bool) else float(classifier_result) >= 0.5

            if has_seatbelt:
                continue

            if not self._should_emit_violation(driver["id"], "no_seatbelt", current_frame):
                continue

            seen_driver_ids.add(driver["id"])
            violations.append(
                {
                    "type": "no_seatbelt",
                    "track_id": driver["id"],
                    "bbox": list(driver["bbox"]),
                    "timestamp": timestamp,
                }
            )

        return violations

    def _default_seatbelt_classifier(self, _driver_crop, _driver: dict, _car: dict, context: dict) -> bool:
        # Demo heuristic: if a detected seatbelt overlaps the driver region, treat as compliant.
        seatbelts = context.get("seatbelts", [])
        driver_region = context.get("driver_region")
        if driver_region is None:
            return False

        return any(
            self._iou(sb["bbox"], driver_region) > 0.02
            or self._point_in_box(self._center(sb["bbox"]), driver_region)
            for sb in seatbelts
        )

    def _update_history(self, tracks: list[dict], timestamp: float) -> None:
        current_ids = {t["id"] for t in tracks}

        for tid, state in self.history.items():
            if tid not in current_ids:
                state.missed_frames += 1

        for track in tracks:
            tid = track["id"]
            cx, cy = self._center(track["bbox"])
            state = self.history.get(tid)
            if state is None:
                state = TrackState(class_name=track["class"], bbox=track["bbox"])
                self.history[tid] = state

            state.class_name = track["class"]
            state.bbox = track["bbox"]
            state.centers.append((cx, cy))
            state.last_seen_ts = timestamp
            state.missed_frames = 0

    def _cleanup_history(self) -> None:
        stale = [tid for tid, state in self.history.items() if state.missed_frames > self.max_missed_frames]
        for tid in stale:
            self.history.pop(tid, None)
            self.detected_violations.pop(tid, None)
            self._violation_last_frame.pop(tid, None)

    def _is_red_light_active(self, tracks: list[dict]) -> bool:
        if self.traffic_light_state is not None:
            return self.traffic_light_state == "red"

        classes = {t["class"] for t in tracks}
        return bool(classes.intersection(self.RED_LABELS))

    def _find_associated_vehicle(
        self,
        person_box: tuple[float, float, float, float],
        motorcycles: list[dict],
        min_iou: float = 0.08,
    ) -> dict | None:
        candidates = [m for m in motorcycles if self._iou(person_box, m["bbox"]) > min_iou]
        if not candidates:
            candidates = [m for m in motorcycles if self._distance(person_box, m["bbox"]) < 90.0]
        if not candidates:
            return None

        return max(candidates, key=lambda m: self._iou(person_box, m["bbox"]))

    def _should_emit_violation(self, track_id: int, violation_type: str, frame_index: int) -> bool:
        types_for_track = self.detected_violations.setdefault(track_id, set())
        last_frame_by_type = self._violation_last_frame.setdefault(track_id, {})
        last_frame = last_frame_by_type.get(violation_type)

        if last_frame is not None and (int(frame_index) - last_frame) < self.violation_cooldown_frames:
            return False

        types_for_track.add(violation_type)
        last_frame_by_type[violation_type] = int(frame_index)
        return True

    def _parse_track(self, raw: dict) -> dict | None:
        if not isinstance(raw, dict):
            return None

        tid = raw.get("id")
        class_name = raw.get("class") or raw.get("class_name")
        bbox = raw.get("bbox")

        if tid is None or class_name is None or bbox is None:
            return None
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return None

        return {
            "id": int(tid),
            "class": str(class_name).lower().strip(),
            "bbox": (x1, y1, x2, y2),
            "confidence": float(raw.get("confidence", 0.0)),
        }

    @staticmethod
    def _extract_head_region(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        return (x1, y1, x2, y1 + 0.25 * (y2 - y1))

    @staticmethod
    def _extract_driver_region(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        return (
            x1,
            y1 + 0.12 * h,
            x1 + 0.45 * w,
            y1 + 0.72 * h,
        )

    @staticmethod
    def _crop_region(frame, box: tuple[float, float, float, float]):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        ix1 = max(0, min(w - 1, int(x1)))
        iy1 = max(0, min(h - 1, int(y1)))
        ix2 = max(ix1 + 1, min(w, int(x2)))
        iy2 = max(iy1 + 1, min(h, int(y2)))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        return frame[iy1:iy2, ix1:ix2]

    @staticmethod
    def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _point_in_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = max(1e-6, area_a + area_b - inter_area)
        return inter_area / union

    @staticmethod
    def _distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax, ay = ViolationEngine._center(a)
        bx, by = ViolationEngine._center(b)
        dx = ax - bx
        dy = ay - by
        return (dx * dx + dy * dy) ** 0.5
