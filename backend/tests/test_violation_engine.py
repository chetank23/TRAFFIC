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

    _ = engine.process_frame(frame_a_tracks, frame, timestamp=1.0)
    violations = engine.process_frame(frame_b_tracks, frame, timestamp=1.2)

    assert any(v["type"] == "red_light" and v["track_id"] == 7 for v in violations)


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
