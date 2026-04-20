from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from detector import Detection
from schemas import BoundingBox, Violation, ViolationType


VIOLATION_LABELS: dict[ViolationType, str] = {
    "no_helmet": "No Helmet",
    "no_seatbelt": "No Seatbelt",
    "red_light": "Red Light Violation",
    "wrong_lane": "Wrong Lane",
    "mobile_usage": "Mobile Phone Usage",
    "overspeeding": "Overspeeding",
    "drunk_driving": "Drunk Driving",
    "no_valid_license": "No Valid License",
    "triple_riding": "Triple Riding",
    "no_parking": "No Parking / Obstruction",
    "dangerous_driving": "Dangerous Driving / Racing",
}

VIOLATION_DESCRIPTIONS: dict[ViolationType, str] = {
    "no_helmet": "Two-wheeler rider detected without helmet indication.",
    "no_seatbelt": "Driver appears unbelted based on model output.",
    "red_light": "Vehicle movement while red traffic light detected.",
    "wrong_lane": "Vehicle appears to be crossing lane direction boundaries.",
    "mobile_usage": "Driver appears to be using a handheld phone.",
    "overspeeding": "Vehicle displacement between frames exceeds speed threshold.",
    "drunk_driving": "Possible intoxicated driving cue detected (strictly heuristic unless model emits explicit drunk-driving class).",
    "no_valid_license": "Driving without valid license indication from model output.",
    "triple_riding": "Two-wheeler appears to carry three or more riders.",
    "no_parking": "Vehicle appears stationary in a no-parking or obstructive road zone.",
    "dangerous_driving": "Aggressive or racing-like motion pattern detected.",
}

MIN_VIOLATION_CONFIDENCE = 0.80

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
TWO_WHEELER_CLASSES = {"motorcycle", "bicycle"}
PHONE_CLASSES = {"cell phone", "mobile phone", "phone"}
NO_HELMET_CLASSES = {"no_helmet", "without_helmet", "helmet_violation", "no-helmet"}
NO_SEATBELT_CLASSES = {"no_seatbelt", "without_seatbelt", "seatbelt_violation", "no-seatbelt"}
WRONG_LANE_CLASSES = {"wrong_lane", "lane_violation"}
RED_LIGHT_CLASSES = {"red_light", "traffic light red", "red signal", "traffic light"}
DRUNK_DRIVING_CLASSES = {"drunk_driving", "drink_and_drive", "dui", "dwi"}
NO_LICENSE_CLASSES = {"no_valid_license", "without_license", "no_driving_license", "license_violation"}
TRIPLE_RIDING_CLASSES = {"triple_riding", "three_on_bike", "overload_two_wheeler"}
NO_PARKING_CLASSES = {"no_parking", "illegal_parking", "parking_violation", "road_obstruction"}
DANGEROUS_DRIVING_CLASSES = {"dangerous_driving", "rash_driving", "racing", "stunt_driving"}
ALCOHOL_CUE_CLASSES = {"wine glass", "bottle", "beer bottle"}


@dataclass
class FrameContext:
    frame_width: int
    frame_height: int
    timestamp: float | None = None
    previous_vehicle_centers: dict[str, tuple[float, float]] | None = None
    stop_line_y_ratio: float = 0.62
    stop_line_band_ratio: float = 0.06


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


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _head_region(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Approximate rider head area from a person box.

    The region is narrowed horizontally and limited to the top section to
    reduce false no-helmet triggers from torso-level overlaps.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    hx1 = x1 + 0.2 * w
    hx2 = x2 - 0.2 * w
    hy1 = y1
    hy2 = y1 + 0.3 * h
    return (hx1, hy1, hx2, hy2)


def _point_in_box(px: float, py: float, box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def _crosses_stop_line(
    prev_y: float,
    curr_y: float,
    line_y: float,
    min_motion_px: float,
) -> bool:
    # Ignore tiny jitter around the line.
    if abs(curr_y - prev_y) < min_motion_px:
        return False
    return (prev_y - line_y) * (curr_y - line_y) <= 0


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


def _new_violation(
    violation_type: ViolationType,
    detection: Detection,
    ctx: FrameContext,
    confidence_boost: float = 0,
    vehicle_id: str | None = None,
) -> Violation:
    confidence = max(0.0, min(1.0, detection.confidence + confidence_boost))
    return Violation(
        id=f"v-{uuid.uuid4().hex[:12]}",
        type=violation_type,
        label=VIOLATION_LABELS[violation_type],
        confidence=confidence,
        timestamp=ctx.timestamp,
        box=_normalize_box(detection.box_xyxy, ctx.frame_width, ctx.frame_height),
        vehicle_id=vehicle_id,
        description=VIOLATION_DESCRIPTIONS[violation_type],
    )


def detect_violations(detections: list[Detection], ctx: FrameContext) -> list[Violation]:
    violations: list[Violation] = []

    by_class: dict[str, list[Detection]] = {}
    for detection in detections:
        by_class.setdefault(detection.class_name, []).append(detection)

    # Model-native explicit labels.
    for cls in NO_HELMET_CLASSES:
        for det in by_class.get(cls, []):
            violations.append(_new_violation("no_helmet", det, ctx))

    for cls in NO_SEATBELT_CLASSES:
        for det in by_class.get(cls, []):
            violations.append(_new_violation("no_seatbelt", det, ctx))

    for cls in WRONG_LANE_CLASSES:
        for det in by_class.get(cls, []):
            violations.append(_new_violation("wrong_lane", det, ctx))

    for cls in DRUNK_DRIVING_CLASSES:
        for det in by_class.get(cls, []):
            violations.append(_new_violation("drunk_driving", det, ctx))

    for cls in NO_LICENSE_CLASSES:
        for det in by_class.get(cls, []):
            violations.append(_new_violation("no_valid_license", det, ctx))

    for cls in TRIPLE_RIDING_CLASSES:
        for det in by_class.get(cls, []):
            violations.append(_new_violation("triple_riding", det, ctx))

    for cls in NO_PARKING_CLASSES:
        for det in by_class.get(cls, []):
            violations.append(_new_violation("no_parking", det, ctx))

    for cls in DANGEROUS_DRIVING_CLASSES:
        for det in by_class.get(cls, []):
            violations.append(_new_violation("dangerous_driving", det, ctx))

    red_light_present = any(by_class.get(cls) for cls in RED_LIGHT_CLASSES)
    stop_line_y = ctx.frame_height * max(0.0, min(1.0, ctx.stop_line_y_ratio))
    stop_line_band = ctx.frame_height * max(0.0, min(0.25, ctx.stop_line_band_ratio))
    min_motion_px = max(8.0, ctx.frame_height * 0.012)

    vehicles = [d for d in detections if d.class_name in VEHICLE_CLASSES]
    vehicle_ids = {id(v): f"VH-{idx + 1:03d}" for idx, v in enumerate(vehicles)}
    riders = [d for d in detections if d.class_name in {"person"}]
    helmets = [d for d in detections if d.class_name in {"helmet", "hardhat"}]
    phones = [d for d in detections if d.class_name in PHONE_CLASSES]
    alcohol_cues = [d for d in detections if d.class_name in ALCOHOL_CUE_CLASSES]

    # Heuristic no-helmet check for two-wheelers with rider-head level checks.
    for vehicle in vehicles:
        if vehicle.class_name not in TWO_WHEELER_CLASSES:
            continue

        # Skip tiny distant detections that tend to create noisy rider inferences.
        vx1, vy1, vx2, vy2 = vehicle.box_xyxy
        vehicle_area_ratio = ((vx2 - vx1) * (vy2 - vy1)) / max(1.0, ctx.frame_width * ctx.frame_height)
        if vehicle_area_ratio < 0.01 or vehicle.confidence < 0.8:
            continue

        overlapping_riders = [r for r in riders if _box_iou(vehicle.box_xyxy, r.box_xyxy) > 0.15]
        if not overlapping_riders:
            continue

        # Use the strongest overlapping rider for stable head-region verification.
        rider = max(overlapping_riders, key=lambda r: _box_iou(vehicle.box_xyxy, r.box_xyxy))
        if rider.confidence < 0.8:
            continue

        head_box = _head_region(rider.box_xyxy)
        rider_has_helmet = False
        for helmet in helmets:
            hx, hy = _box_center(helmet.box_xyxy)
            if _box_iou(head_box, helmet.box_xyxy) > 0.03 or _point_in_box(hx, hy, head_box):
                rider_has_helmet = True
                break

        if not rider_has_helmet:
            violations.append(
                _new_violation(
                    "no_helmet",
                    vehicle,
                    ctx,
                    confidence_boost=-0.05,
                    vehicle_id=vehicle_ids.get(id(vehicle)),
                )
            )

        # If multiple riders are detected and helmet count is lower than rider count,
        # flag helmet non-compliance (covers pillion rider without helmet).
        helmeted_riders = 0
        for rider_candidate in overlapping_riders:
            rider_head = _head_region(rider_candidate.box_xyxy)
            has_helmet = any(_box_iou(rider_head, h.box_xyxy) > 0.03 for h in helmets)
            helmeted_riders += 1 if has_helmet else 0

        if len(overlapping_riders) >= 2 and helmeted_riders < len(overlapping_riders):
            violations.append(
                _new_violation(
                    "no_helmet",
                    vehicle,
                    ctx,
                    confidence_boost=-0.02,
                    vehicle_id=vehicle_ids.get(id(vehicle)),
                )
            )

        if len(overlapping_riders) >= 3:
            violations.append(
                _new_violation(
                    "triple_riding",
                    vehicle,
                    ctx,
                    confidence_boost=-0.02,
                    vehicle_id=vehicle_ids.get(id(vehicle)),
                )
            )

    # No-seatbelt is only emitted from explicit detector labels to avoid false positives
    # when using generic YOLO classes without occupant-seatbelt visibility cues.

    # Phone usage: phone and person close with overlap.
    for person in riders:
        nearby_phone = any(_box_iou(person.box_xyxy, phone.box_xyxy) > 0.06 for phone in phones)
        if nearby_phone:
            violations.append(_new_violation("mobile_usage", person, ctx, confidence_boost=-0.03))

    # Drunk-driving heuristic (non-conclusive): alcohol object overlapping driver/rider area.
    for person in riders:
        near_alcohol = any(_box_iou(person.box_xyxy, cue.box_xyxy) > 0.06 for cue in alcohol_cues)
        if near_alcohol:
            violations.append(_new_violation("drunk_driving", person, ctx, confidence_boost=-0.02))

    # Red light + zebra/stop-line crossing rule.
    if red_light_present:
        for idx, vehicle in enumerate(vehicles):
            _, cy = _box_center(vehicle.box_xyxy)
            key = f"{vehicle.class_name}-{idx}"
            prev_center = ctx.previous_vehicle_centers.get(key) if ctx.previous_vehicle_centers else None

            crossed_line = False
            if prev_center is not None:
                _, prev_y = prev_center
                crossed_line = _crosses_stop_line(prev_y, cy, stop_line_y, min_motion_px)

            on_line_zone = abs(cy - stop_line_y) <= stop_line_band
            if crossed_line or on_line_zone:
                violations.append(
                    _new_violation(
                        "red_light",
                        vehicle,
                        ctx,
                        confidence_boost=0.02,
                        vehicle_id=vehicle_ids.get(id(vehicle)),
                    )
                )

    # Wrong lane fallback: vehicle significantly in extreme left/right regions.
    for vehicle in vehicles:
        cx, _ = _box_center(vehicle.box_xyxy)
        if cx < ctx.frame_width * 0.16 or cx > ctx.frame_width * 0.84:
            violations.append(
                _new_violation(
                    "wrong_lane",
                    vehicle,
                    ctx,
                    confidence_boost=-0.12,
                    vehicle_id=vehicle_ids.get(id(vehicle)),
                )
            )

    # Overspeeding: significant frame-to-frame displacement of the same indexed vehicle.
    overspeed_keys: set[str] = set()
    if ctx.previous_vehicle_centers:
        for idx, vehicle in enumerate(vehicles):
            key = f"{vehicle.class_name}-{idx}"
            prev_center = ctx.previous_vehicle_centers.get(key)
            if prev_center is None:
                continue

            vx, vy = _box_center(vehicle.box_xyxy)
            dist = math.dist(prev_center, (vx, vy))
            frame_diag = math.sqrt(ctx.frame_width**2 + ctx.frame_height**2)
            normalized_disp = dist / max(frame_diag, 1)

            if normalized_disp > 0.11:
                overspeed_keys.add(key)
                violations.append(
                    _new_violation(
                        "overspeeding",
                        vehicle,
                        ctx,
                        confidence_boost=-0.02,
                        vehicle_id=vehicle_ids.get(id(vehicle)),
                    )
                )

            # Stationary obstructive vehicle heuristic (for no-parking/middle-road context).
            if normalized_disp < 0.008:
                cx, cy = _box_center(vehicle.box_xyxy)
                in_middle_road_zone = (
                    ctx.frame_width * 0.35 <= cx <= ctx.frame_width * 0.65
                    and cy >= ctx.frame_height * 0.38
                )
                if in_middle_road_zone and not red_light_present:
                    violations.append(
                        _new_violation(
                            "no_parking",
                            vehicle,
                            ctx,
                            confidence_boost=-0.05,
                            vehicle_id=vehicle_ids.get(id(vehicle)),
                        )
                    )

    # Dangerous driving / racing heuristic from high displacement combined with risky trajectory.
    for idx, vehicle in enumerate(vehicles):
        key = f"{vehicle.class_name}-{idx}"
        if key not in overspeed_keys:
            continue
        cx, _ = _box_center(vehicle.box_xyxy)
        risky_lane_position = cx < ctx.frame_width * 0.16 or cx > ctx.frame_width * 0.84
        if risky_lane_position or red_light_present:
            violations.append(
                _new_violation(
                    "dangerous_driving",
                    vehicle,
                    ctx,
                    confidence_boost=-0.01,
                    vehicle_id=vehicle_ids.get(id(vehicle)),
                )
            )

    # De-duplicate by type and approximate same box region.
    deduped: list[Violation] = []
    for violation in sorted(violations, key=lambda v: v.confidence, reverse=True):
        is_duplicate = False
        for existing in deduped:
            if violation.type != existing.type:
                continue
            if abs(violation.box.x - existing.box.x) < 0.03 and abs(violation.box.y - existing.box.y) < 0.03:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(violation)

    return [v for v in deduped if v.confidence >= MIN_VIOLATION_CONFIDENCE]
