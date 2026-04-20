from detector import Detection
from violations import FrameContext, detect_violations


def test_detects_mobile_usage_from_phone_overlap():
    detections = [
        Detection(0, "person", 0.93, (100, 100, 220, 300)),
        Detection(67, "cell phone", 0.88, (180, 180, 240, 280)),
    ]
    ctx = FrameContext(frame_width=640, frame_height=480, timestamp=1.5)

    violations = detect_violations(detections, ctx)
    assert any(v.type == "mobile_usage" for v in violations)


def test_detects_no_helmet_for_rider_without_helmet_detection():
    detections = [
        Detection(3, "motorcycle", 0.9, (140, 180, 360, 420)),
        Detection(0, "person", 0.89, (190, 150, 320, 390)),
    ]
    ctx = FrameContext(frame_width=800, frame_height=600)

    violations = detect_violations(detections, ctx)
    assert any(v.type == "no_helmet" for v in violations)


def test_red_light_violation_when_crossing_stop_line():
    detections = [
        Detection(2, "car", 0.95, (250, 250, 430, 410)),
        Detection(9, "traffic light", 0.91, (40, 35, 90, 145)),
    ]
    ctx = FrameContext(
        frame_width=640,
        frame_height=480,
        timestamp=1.8,
        previous_vehicle_centers={"car-0": (340, 220)},
        stop_line_y_ratio=0.62,
        stop_line_band_ratio=0.05,
    )

    violations = detect_violations(detections, ctx)
    assert any(v.type == "red_light" for v in violations)


def test_no_red_light_without_signal():
    detections = [
        Detection(2, "car", 0.95, (250, 250, 430, 410)),
    ]
    ctx = FrameContext(
        frame_width=640,
        frame_height=480,
        timestamp=1.8,
        previous_vehicle_centers={"car-0": (340, 220)},
        stop_line_y_ratio=0.62,
        stop_line_band_ratio=0.05,
    )

    violations = detect_violations(detections, ctx)
    assert not any(v.type == "red_light" for v in violations)


def test_detects_overspeeding_from_large_displacement():
    detections = [
        Detection(2, "car", 0.93, (300, 220, 430, 360)),
    ]
    ctx = FrameContext(
        frame_width=640,
        frame_height=480,
        timestamp=2.2,
        previous_vehicle_centers={"car-0": (120, 220)},
    )

    violations = detect_violations(detections, ctx)
    assert any(v.type == "overspeeding" for v in violations)


def test_detects_triple_riding_on_two_wheeler():
    detections = [
        Detection(3, "motorcycle", 0.95, (130, 180, 370, 430)),
        Detection(0, "person", 0.92, (165, 150, 255, 390)),
        Detection(0, "person", 0.91, (235, 155, 315, 390)),
        Detection(0, "person", 0.90, (300, 160, 360, 390)),
    ]
    ctx = FrameContext(frame_width=800, frame_height=600, timestamp=3.0)

    violations = detect_violations(detections, ctx)
    assert any(v.type == "triple_riding" for v in violations)


def test_detects_no_valid_license_from_explicit_label():
    detections = [
        Detection(501, "no_valid_license", 0.91, (120, 120, 260, 340)),
    ]
    ctx = FrameContext(frame_width=640, frame_height=480, timestamp=1.0)

    violations = detect_violations(detections, ctx)
    assert any(v.type == "no_valid_license" for v in violations)


def test_detects_no_parking_for_stationary_middle_road_vehicle():
    detections = [
        Detection(2, "car", 0.9, (260, 210, 420, 380)),
    ]
    ctx = FrameContext(
        frame_width=640,
        frame_height=480,
        timestamp=4.2,
        previous_vehicle_centers={"car-0": (340, 295)},
    )

    violations = detect_violations(detections, ctx)
    assert any(v.type == "no_parking" for v in violations)


def test_detects_dangerous_driving_from_overspeed_and_risky_lane():
    detections = [
        Detection(2, "car", 0.93, (10, 210, 150, 350)),
    ]
    ctx = FrameContext(
        frame_width=640,
        frame_height=480,
        timestamp=5.1,
        previous_vehicle_centers={"car-0": (320, 280)},
    )

    violations = detect_violations(detections, ctx)
    assert any(v.type == "dangerous_driving" for v in violations)


def test_detects_drunk_driving_from_alcohol_overlap_cue():
    detections = [
        Detection(0, "person", 0.93, (180, 120, 320, 380)),
        Detection(39, "bottle", 0.9, (235, 220, 295, 330)),
    ]
    ctx = FrameContext(frame_width=640, frame_height=480, timestamp=2.4)

    violations = detect_violations(detections, ctx)
    assert any(v.type == "drunk_driving" for v in violations)
