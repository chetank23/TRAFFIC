import numpy as np

from violation_engine import ViolationEngine


def test_no_helmet_rule_emits_violation_for_rider_without_helmet():
    engine = ViolationEngine(stop_line_ratio=0.62)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = [
        {"id": 1, "class": "person", "bbox": [300, 220, 420, 620], "confidence": 0.9},
        {"id": 2, "class": "motorcycle", "bbox": [250, 360, 520, 700], "confidence": 0.92},
    ]

    violations = engine.process_frame(tracks, frame, timestamp=1.0)

    assert any(v["type"] == "no_helmet" and v["track_id"] == 1 for v in violations)


def test_no_helmet_rule_ignores_when_helmet_overlaps_head_region():
    engine = ViolationEngine(stop_line_ratio=0.62)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = [
        {"id": 1, "class": "person", "bbox": [300, 220, 420, 620], "confidence": 0.9},
        {"id": 2, "class": "motorcycle", "bbox": [250, 360, 520, 700], "confidence": 0.92},
        {"id": 3, "class": "helmet", "bbox": [320, 230, 380, 300], "confidence": 0.87},
    ]

    violations = engine.process_frame(tracks, frame, timestamp=1.0)

    assert not any(v["type"] == "no_helmet" and v["track_id"] == 1 for v in violations)


def test_no_helmet_rule_handles_multiple_riders():
    engine = ViolationEngine(stop_line_ratio=0.62)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = [
        {"id": 10, "class": "person", "bbox": [280, 220, 380, 610], "confidence": 0.9},
        {"id": 11, "class": "person", "bbox": [380, 230, 470, 620], "confidence": 0.9},
        {"id": 20, "class": "motorcycle", "bbox": [240, 340, 520, 700], "confidence": 0.93},
        {"id": 30, "class": "helmet", "bbox": [300, 230, 360, 290], "confidence": 0.88},
    ]

    violations = engine.process_frame(tracks, frame, timestamp=2.0)

    assert any(v["type"] == "no_helmet" and v["track_id"] == 11 for v in violations)
    assert not any(v["type"] == "no_helmet" and v["track_id"] == 10 for v in violations)


def test_no_helmet_rule_suppresses_duplicates_for_same_track_in_cooldown():
    engine = ViolationEngine(stop_line_ratio=0.62)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = [
        {"id": 1, "class": "person", "bbox": [300, 220, 420, 620], "confidence": 0.9},
        {"id": 2, "class": "motorcycle", "bbox": [250, 360, 520, 700], "confidence": 0.92},
    ]

    violations_a = engine.process_frame(tracks, frame, timestamp=3.0)
    violations_b = engine.process_frame(tracks, frame, timestamp=3.3)

    assert any(v["type"] == "no_helmet" and v["track_id"] == 1 for v in violations_a)
    assert not any(v["type"] == "no_helmet" and v["track_id"] == 1 for v in violations_b)


def test_dedup_dictionary_records_violation_types_per_track():
    engine = ViolationEngine(stop_line_ratio=0.62)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = [
        {"id": 1, "class": "person", "bbox": [300, 220, 420, 620], "confidence": 0.9},
        {"id": 2, "class": "motorcycle", "bbox": [250, 360, 520, 700], "confidence": 0.92},
    ]

    _ = engine.process_frame(tracks, frame, timestamp=3.0, frame_index=90)

    assert 1 in engine.detected_violations
    assert "no_helmet" in engine.detected_violations[1]


def test_same_violation_allowed_again_after_cooldown_frames():
    engine = ViolationEngine(stop_line_ratio=0.62, violation_cooldown_frames=5)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = [
        {"id": 1, "class": "person", "bbox": [300, 220, 420, 620], "confidence": 0.9},
        {"id": 2, "class": "motorcycle", "bbox": [250, 360, 520, 700], "confidence": 0.92},
    ]

    first = engine.process_frame(tracks, frame, frame_index=10)
    blocked = engine.process_frame(tracks, frame, frame_index=12)
    allowed = engine.process_frame(tracks, frame, frame_index=16)

    assert any(v["type"] == "no_helmet" and v["track_id"] == 1 for v in first)
    assert not any(v["type"] == "no_helmet" and v["track_id"] == 1 for v in blocked)
    assert any(v["type"] == "no_helmet" and v["track_id"] == 1 for v in allowed)


def test_red_light_rule_emits_on_stop_line_crossing():
    engine = ViolationEngine(stop_line_ratio=0.62)
    engine.set_traffic_light_state("red")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    frame_a_tracks = [
        {"id": 7, "class": "car", "bbox": [280, 180, 360, 230], "confidence": 0.9},
    ]
    frame_b_tracks = [
        {"id": 7, "class": "car", "bbox": [280, 320, 360, 420], "confidence": 0.91},
    ]

    _ = engine.process_frame(frame_a_tracks, frame, timestamp=1.0, frame_index=30)
    violations = engine.process_frame(frame_b_tracks, frame, timestamp=1.2, frame_index=36)

    assert any(
        v["type"] == "red_light_violation" and v["track_id"] == 7 and v.get("frame") == 36
        for v in violations
    )


def test_red_light_rule_logs_only_once_per_track():
    engine = ViolationEngine(stop_line_ratio=0.62)
    engine.set_traffic_light_state("red")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    frame_a_tracks = [{"id": 5, "class": "car", "bbox": [260, 160, 360, 220], "confidence": 0.9}]
    frame_b_tracks = [{"id": 5, "class": "car", "bbox": [260, 300, 360, 410], "confidence": 0.91}]
    frame_c_tracks = [{"id": 5, "class": "car", "bbox": [260, 310, 360, 420], "confidence": 0.92}]

    _ = engine.process_frame(frame_a_tracks, frame, frame_index=1)
    first_cross = engine.process_frame(frame_b_tracks, frame, frame_index=2)
    second_cross = engine.process_frame(frame_c_tracks, frame, frame_index=3)

    assert any(v["type"] == "red_light_violation" and v["track_id"] == 5 for v in first_cross)
    assert not any(v["type"] == "red_light_violation" and v["track_id"] == 5 for v in second_cross)


def test_lost_tracks_removed_after_missed_threshold():
    engine = ViolationEngine(max_missed_frames=2)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    _ = engine.process_frame(
        [{"id": 99, "class": "car", "bbox": [10, 10, 80, 80], "confidence": 0.8}],
        frame,
        timestamp=0.0,
    )
    _ = engine.process_frame([], frame, timestamp=0.1)
    _ = engine.process_frame([], frame, timestamp=0.2)
    _ = engine.process_frame([], frame, timestamp=0.3)

    assert 99 not in engine.history


def test_no_seatbelt_rule_emits_violation_for_driver_without_seatbelt():
    engine = ViolationEngine(stop_line_ratio=0.62)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = [
        {"id": 101, "class": "car", "bbox": [150, 260, 760, 700], "confidence": 0.93},
        {"id": 7, "class": "person", "bbox": [180, 320, 430, 680], "confidence": 0.9},
    ]

    violations = engine.detect_no_seatbelt_violations(tracks, frame, timestamp=4.0)

    assert any(v["type"] == "no_seatbelt" and v["track_id"] == 7 for v in violations)


def test_no_seatbelt_rule_skips_when_seatbelt_detected_in_driver_region():
    engine = ViolationEngine(stop_line_ratio=0.62)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = [
        {"id": 101, "class": "car", "bbox": [150, 260, 760, 700], "confidence": 0.93},
        {"id": 7, "class": "person", "bbox": [180, 320, 430, 680], "confidence": 0.9},
        {"id": 55, "class": "seatbelt", "bbox": [210, 360, 290, 460], "confidence": 0.82},
    ]

    violations = engine.detect_no_seatbelt_violations(tracks, frame, timestamp=4.0)

    assert not any(v["type"] == "no_seatbelt" and v["track_id"] == 7 for v in violations)


def test_no_seatbelt_rule_supports_custom_classifier_hook():
    engine = ViolationEngine(stop_line_ratio=0.62)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = [
        {"id": 101, "class": "car", "bbox": [150, 260, 760, 700], "confidence": 0.93},
        {"id": 7, "class": "person", "bbox": [180, 320, 430, 680], "confidence": 0.9},
    ]

    def always_has_seatbelt(_crop, _driver, _car, _context):
        return True

    engine.set_seatbelt_classifier(always_has_seatbelt)
    violations = engine.detect_no_seatbelt_violations(tracks, frame, timestamp=4.0)

    assert not any(v["type"] == "no_seatbelt" and v["track_id"] == 7 for v in violations)


def test_no_seatbelt_rule_avoids_duplicates_for_same_track_in_cooldown():
    engine = ViolationEngine(stop_line_ratio=0.62)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = [
        {"id": 101, "class": "car", "bbox": [150, 260, 760, 700], "confidence": 0.93},
        {"id": 7, "class": "person", "bbox": [180, 320, 430, 680], "confidence": 0.9},
    ]

    violations_a = engine.detect_no_seatbelt_violations(tracks, frame, timestamp=5.0)
    violations_b = engine.detect_no_seatbelt_violations(tracks, frame, timestamp=5.2)

    assert any(v["type"] == "no_seatbelt" and v["track_id"] == 7 for v in violations_a)
    assert not any(v["type"] == "no_seatbelt" and v["track_id"] == 7 for v in violations_b)
