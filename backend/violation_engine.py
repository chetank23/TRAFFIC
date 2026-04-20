from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


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

    def __init__(self, stop_line_ratio: float = 0.62, max_missed_frames: int = 25) -> None:
        self.history: dict[int, TrackState] = {}
        self.stop_line_ratio = stop_line_ratio
        self.max_missed_frames = max_missed_frames
        self.traffic_light_state: str | None = None
        self._violation_cooldown: dict[tuple[int, str], float] = {}

    def set_traffic_light_state(self, state: str | None) -> None:
        self.traffic_light_state = state.lower().strip() if state else None

    def process_frame(self, tracks: list[dict], frame, timestamp: float | None = None) -> list[dict]:
        ts = float(timestamp) if timestamp is not None else time.time()
        frame_h = frame.shape[0] if frame is not None else 0
        stop_line_y = frame_h * max(0.0, min(1.0, self.stop_line_ratio))

        parsed_tracks = [t for t in (self._parse_track(x) for x in tracks) if t is not None]
        self._update_history(parsed_tracks, ts)

        violations: list[dict] = []
        violations.extend(self._check_helmet_rule(parsed_tracks, ts))
        violations.extend(self._check_red_light_rule(parsed_tracks, stop_line_y, ts))
        violations.extend(self._check_seatbelt_placeholder(parsed_tracks, ts))

        self._cleanup_history()
        return violations

    def _check_helmet_rule(self, tracks: list[dict], timestamp: float) -> list[dict]:
        people = [t for t in tracks if t["class"] == "person"]
        helmets = [t for t in tracks if t["class"] == "helmet"]
        motorcycles = [t for t in tracks if t["class"] == "motorcycle"]

        violations: list[dict] = []
        for person in people:
            rider_bike = self._find_associated_vehicle(person["bbox"], motorcycles)
            if rider_bike is None:
                continue

            head_box = self._extract_head_region(person["bbox"])
            has_helmet = any(
                self._iou(head_box, helmet["bbox"]) > 0.03
                or self._point_in_box(self._center(helmet["bbox"]), head_box)
                for helmet in helmets
            )

            if has_helmet:
                continue

            if self._cooldown_active(person["id"], "no_helmet", timestamp, 1.4):
                continue

            violations.append(
                {
                    "track_id": person["id"],
                    "type": "no_helmet",
                    "bbox": list(person["bbox"]),
                    "timestamp": timestamp,
                }
            )

        return violations

    def _check_red_light_rule(self, tracks: list[dict], stop_line_y: float, timestamp: float) -> list[dict]:
        red_active = self._is_red_light_active(tracks)
        if not red_active:
            return []

        violations: list[dict] = []
        for track in tracks:
            if track["class"] not in self.VEHICLE_CLASSES:
                continue

            state = self.history.get(track["id"])
            if state is None or len(state.centers) < 2:
                continue

            prev_y = state.centers[-2][1]
            curr_y = state.centers[-1][1]
            crossed = (prev_y - stop_line_y) * (curr_y - stop_line_y) <= 0 and abs(curr_y - prev_y) > 5

            if not crossed:
                continue

            if self._cooldown_active(track["id"], "red_light", timestamp, 1.8):
                continue

            violations.append(
                {
                    "track_id": track["id"],
                    "type": "red_light",
                    "bbox": list(track["bbox"]),
                    "timestamp": timestamp,
                }
            )

        return violations

    def _check_seatbelt_placeholder(self, tracks: list[dict], timestamp: float) -> list[dict]:
        # Placeholder for seatbelt-specific classifier or richer pose cues.
        # Keep the hook modular so the rule can be upgraded without touching other rules.
        _ = tracks
        _ = timestamp
        return []

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

    def _is_red_light_active(self, tracks: list[dict]) -> bool:
        if self.traffic_light_state is not None:
            return self.traffic_light_state == "red"

        classes = {t["class"] for t in tracks}
        return bool(classes.intersection(self.RED_LABELS))

    def _find_associated_vehicle(
        self, person_box: tuple[float, float, float, float], motorcycles: list[dict]
    ) -> dict | None:
        candidates = [m for m in motorcycles if self._iou(person_box, m["bbox"]) > 0.08]
        if not candidates:
            candidates = [m for m in motorcycles if self._distance(person_box, m["bbox"]) < 90.0]
        if not candidates:
            return None

        return max(candidates, key=lambda m: self._iou(person_box, m["bbox"]))

    def _cooldown_active(self, track_id: int, violation_type: str, timestamp: float, seconds: float) -> bool:
        key = (track_id, violation_type)
        last_ts = self._violation_cooldown.get(key)
        if last_ts is not None and (timestamp - last_ts) < seconds:
            return True

        self._violation_cooldown[key] = timestamp
        return False

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
