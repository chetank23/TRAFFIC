import numpy as np

from violation_detector import ViolationDetector


def test_no_helmet_violation_detected_for_rider():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = [
        {"class_name": "person", "confidence": 0.92, "bbox": [300, 220, 420, 620]},
        {"class_name": "motorcycle", "confidence": 0.9, "bbox": [260, 360, 500, 700]},
    ]

    detector = ViolationDetector()
    violations = detector.detect_violations(detections, frame)

    assert any(v["type"] == "no_helmet" for v in violations)


def test_no_seatbelt_violation_detected_for_driver_region():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = [
        {"class_name": "car", "confidence": 0.91, "bbox": [150, 250, 700, 650]},
        {"class_name": "person", "confidence": 0.89, "bbox": [180, 320, 380, 620]},
    ]

    detector = ViolationDetector()
    violations = detector.detect_violations(detections, frame)

    assert any(v["type"] == "no_seatbelt" for v in violations)


def test_red_light_violation_detected_when_crossing_stop_line():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = ViolationDetector(stop_line_ratio=0.62)

    first_frame = [
        {"class_name": "car", "confidence": 0.9, "bbox": [250, 170, 380, 250]},
    ]
    second_frame = [
        {"class_name": "car", "confidence": 0.92, "bbox": [250, 300, 380, 410]},
    ]

    _ = detector.detect_violations(first_frame, frame, traffic_light_state="red")
    violations = detector.detect_violations(second_frame, frame, traffic_light_state="red")

    assert any(v["type"] == "red_light_violation" for v in violations)


def test_draw_violations_adds_annotations():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = ViolationDetector()

    violations = [
        {"type": "no_helmet", "bbox": [50, 50, 180, 180], "confidence": 0.88},
    ]

    annotated = detector.draw_violations(frame, violations)

    assert annotated.shape == frame.shape
    assert annotated.sum() > frame.sum()
