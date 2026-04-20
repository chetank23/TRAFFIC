from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class ParsedDetection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


class ViolationDetector:
    """Rule-based traffic violation detector built on top of YOLO detections.

    Input detections can be objects (with class_name/confidence/box_xyxy) or dictionaries
    with equivalent fields.
    """

    VEHICLE_CLASSES = {"car", "motorcycle"}
    RED_LIGHT_LABELS = {"red", "red_light", "traffic light red", "red signal"}

    def __init__(self, stop_line_ratio: float = 0.62) -> None:
        self.stop_line_ratio = stop_line_ratio
        self._previous_vehicle_centers: dict[str, tuple[float, float]] = {}

    def detect_violations(
        self,
        detections: list[Any],
        frame: np.ndarray,
        traffic_light_state: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return violation dictionaries for a single frame.

        Output format:
        [
          {
            "type": "no_helmet",
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.91
          }
        ]
        """
        parsed = self._parse_detections(detections)
        if frame is None or frame.size == 0:
            return []

        frame_h, frame_w = frame.shape[:2]

        violations: list[dict[str, Any]] = []
        violations.extend(self.check_no_helmet(parsed))
        violations.extend(self.check_no_seatbelt(parsed))
        violations.extend(
            self.check_red_light(
                parsed,
                frame_width=frame_w,
                frame_height=frame_h,
                traffic_light_state=traffic_light_state,
            )
        )

        return self._dedupe_violations(violations)

    def check_no_helmet(self, detections: list[ParsedDetection]) -> list[dict[str, Any]]:
        people = [d for d in detections if d.class_name == "person"]
        motorcycles = [d for d in detections if d.class_name == "motorcycle"]
        helmets = [d for d in detections if d.class_name == "helmet"]

        violations: list[dict[str, Any]] = []

        for person in people:
            rider_bike = self._find_associated_vehicle(person.bbox, motorcycles)
            if rider_bike is None:
                continue

            head_region = self.extract_head_region(person.bbox)
            has_helmet = any(
                self.iou(head_region, helmet.bbox) > 0.02
                or self._point_in_box(self._box_center(helmet.bbox), head_region)
                for helmet in helmets
            )

            if not has_helmet:
                confidence = max(0.0, min(1.0, 0.55 * person.confidence + 0.45 * rider_bike.confidence))
                violations.append(
                    {
                        "type": "no_helmet",
                        "bbox": self._to_int_box(person.bbox),
                        "confidence": round(confidence, 4),
                    }
                )

        return violations

    def check_no_seatbelt(self, detections: list[ParsedDetection]) -> list[dict[str, Any]]:
        people = [d for d in detections if d.class_name == "person"]
        cars = [d for d in detections if d.class_name == "car"]
        seatbelts = [d for d in detections if d.class_name == "seatbelt"]

        violations: list[dict[str, Any]] = []

        for car in cars:
            driver_region = self.extract_driver_region(car.bbox)

            driver_candidates = [
                person
                for person in people
                if self.iou(person.bbox, car.bbox) > 0.05
                and (
                    self.iou(person.bbox, driver_region) > 0.03
                    or self._point_in_box(self._box_center(person.bbox), driver_region)
                )
            ]

            if not driver_candidates:
                continue

            seatbelt_present = any(
                self.iou(sb.bbox, driver_region) > 0.02
                or self._point_in_box(self._box_center(sb.bbox), driver_region)
                for sb in seatbelts
            )

            if not seatbelt_present:
                driver = max(driver_candidates, key=lambda p: p.confidence)
                confidence = max(0.0, min(1.0, 0.5 * driver.confidence + 0.5 * car.confidence))
                violations.append(
                    {
                        "type": "no_seatbelt",
                        "bbox": self._to_int_box(driver.bbox),
                        "confidence": round(confidence, 4),
                    }
                )

        return violations

    def check_red_light(
        self,
        detections: list[ParsedDetection],
        frame_width: int,
        frame_height: int,
        traffic_light_state: str | None = None,
    ) -> list[dict[str, Any]]:
        red_active = self._is_red_light_active(detections, traffic_light_state)
        vehicles = [d for d in detections if d.class_name in self.VEHICLE_CLASSES]

        stop_line_y = frame_height * max(0.0, min(1.0, self.stop_line_ratio))
        sorted_vehicles = sorted(vehicles, key=lambda d: (d.class_name, self._box_center(d.bbox)[0]))

        violations: list[dict[str, Any]] = []
        updated_centers: dict[str, tuple[float, float]] = {}

        for idx, vehicle in enumerate(sorted_vehicles):
            cx, cy = self._box_center(vehicle.bbox)
            key = f"{vehicle.class_name}:{idx}"
            updated_centers[key] = (cx, cy)

            prev_center = self._previous_vehicle_centers.get(key)
            crossed = False
            if prev_center is not None:
                prev_y = prev_center[1]
                crossed = (prev_y - stop_line_y) * (cy - stop_line_y) <= 0 and abs(cy - prev_y) > 6

            on_or_beyond_line = cy >= stop_line_y
            if red_active and (crossed or (prev_center is None and on_or_beyond_line)):
                violations.append(
                    {
                        "type": "red_light_violation",
                        "bbox": self._to_int_box(vehicle.bbox),
                        "confidence": round(vehicle.confidence, 4),
                    }
                )

        self._previous_vehicle_centers = updated_centers
        return violations

    @staticmethod
    def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area <= 0:
            return 0.0

        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union

    @staticmethod
    def is_spatially_close(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
        max_distance_px: float = 70.0,
        min_iou: float = 0.01,
    ) -> bool:
        if ViolationDetector.iou(box_a, box_b) >= min_iou:
            return True

        ax, ay = ViolationDetector._box_center(box_a)
        bx, by = ViolationDetector._box_center(box_b)
        dx = ax - bx
        dy = ay - by
        return (dx * dx + dy * dy) ** 0.5 <= max_distance_px

    @staticmethod
    def extract_head_region(person_box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = person_box
        h = y2 - y1
        return (x1, y1, x2, y1 + 0.25 * h)

    @staticmethod
    def extract_driver_region(car_box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = car_box
        w = x2 - x1
        h = y2 - y1

        # Front-left approximation for right-hand traffic camera assumptions.
        rx1 = x1
        rx2 = x1 + 0.45 * w
        ry1 = y1 + 0.15 * h
        ry2 = y1 + 0.70 * h
        return (rx1, ry1, rx2, ry2)

    def draw_violations(
        self,
        frame: np.ndarray,
        violations: list[dict[str, Any]],
    ) -> np.ndarray:
        color_map = {
            "no_helmet": (0, 0, 255),
            "no_seatbelt": (0, 165, 255),
            "red_light_violation": (255, 0, 255),
        }

        output = frame.copy()
        for item in violations:
            x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
            violation_type = str(item["type"])
            confidence = float(item["confidence"])
            color = color_map.get(violation_type, (255, 255, 0))

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{violation_type} {confidence:.2f}"
            cv2.putText(
                output,
                label,
                (x1, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return output

    def _find_associated_vehicle(
        self,
        person_box: tuple[float, float, float, float],
        vehicles: list[ParsedDetection],
    ) -> ParsedDetection | None:
        candidates = [v for v in vehicles if self.is_spatially_close(person_box, v.bbox)]
        if not candidates:
            return None
        return max(candidates, key=lambda v: self.iou(person_box, v.bbox))

    def _is_red_light_active(
        self,
        detections: list[ParsedDetection],
        traffic_light_state: str | None,
    ) -> bool:
        if traffic_light_state is not None:
            return traffic_light_state.strip().lower() == "red"

        detected_classes = {d.class_name for d in detections}
        if "traffic light" in detected_classes:
            # Generic "traffic light" class does not encode state by itself.
            return False

        return bool(detected_classes.intersection(self.RED_LIGHT_LABELS))

    def _parse_detections(self, detections: list[Any]) -> list[ParsedDetection]:
        parsed: list[ParsedDetection] = []
        for item in detections:
            obj = self._parse_detection(item)
            if obj is not None:
                parsed.append(obj)
        return parsed

    def _parse_detection(self, item: Any) -> ParsedDetection | None:
        if isinstance(item, ParsedDetection):
            return item

        class_name: str | None = None
        confidence: float | None = None
        bbox: tuple[float, float, float, float] | None = None

        if isinstance(item, dict):
            class_name = (
                item.get("class_name")
                or item.get("class")
                or item.get("label")
                or item.get("name")
            )
            confidence = item.get("confidence") or item.get("conf")
            raw_box = item.get("bbox") or item.get("box") or item.get("xyxy")
            bbox = self._to_box_tuple(raw_box)
        else:
            class_name = getattr(item, "class_name", None) or getattr(item, "label", None)
            confidence = getattr(item, "confidence", None) or getattr(item, "conf", None)
            raw_box = getattr(item, "bbox", None) or getattr(item, "box_xyxy", None)
            bbox = self._to_box_tuple(raw_box)

        if not class_name or confidence is None or bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None

        return ParsedDetection(
            class_name=str(class_name).strip().lower(),
            confidence=float(confidence),
            bbox=(float(x1), float(y1), float(x2), float(y2)),
        )

    @staticmethod
    def _to_box_tuple(raw_box: Any) -> tuple[float, float, float, float] | None:
        if raw_box is None:
            return None

        if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            return (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))

        # Supports tensor-like boxes where first item contains [x1, y1, x2, y2]
        if hasattr(raw_box, "tolist"):
            values = raw_box.tolist()
            if isinstance(values, list) and len(values) == 4:
                return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
            if isinstance(values, list) and len(values) > 0 and isinstance(values[0], list) and len(values[0]) == 4:
                first = values[0]
                return (float(first[0]), float(first[1]), float(first[2]), float(first[3]))

        return None

    @staticmethod
    def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _point_in_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
        px, py = point
        x1, y1, x2, y2 = box
        return x1 <= px <= x2 and y1 <= py <= y2

    @staticmethod
    def _to_int_box(box: tuple[float, float, float, float]) -> list[int]:
        x1, y1, x2, y2 = box
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    def _dedupe_violations(self, violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        for item in sorted(violations, key=lambda x: float(x["confidence"]), reverse=True):
            bbox = tuple(item["bbox"])
            vtype = str(item["type"])

            duplicate = False
            for existing in deduped:
                if vtype != str(existing["type"]):
                    continue
                if self.iou(bbox, tuple(existing["bbox"])) > 0.65:
                    duplicate = True
                    break

            if not duplicate:
                deduped.append(item)

        return deduped
