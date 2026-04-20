from __future__ import annotations

import numpy as np

from evidence import EvidenceCapture


def test_capture_writes_image_and_metadata(tmp_path):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[20:90, 30:120] = 255

    capture = EvidenceCapture(root_dir=str(tmp_path), jpeg_quality=70, max_files=50)
    violations = [
        {"type": "no_helmet", "track_id": 8, "timestamp": 10.25},
    ]

    metadata = capture.capture(frame, violations)

    assert len(metadata) == 1
    assert metadata[0]["type"] == "no_helmet"
    assert "violation_no_helmet_8_10_250.jpg" in metadata[0]["image_path"]


def test_capture_skips_duplicate_frame_for_same_track_and_type(tmp_path):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[10:100, 20:130] = 200

    capture = EvidenceCapture(root_dir=str(tmp_path), jpeg_quality=70, max_files=50)
    violations_a = [
        {"type": "no_seatbelt", "track_id": 3, "timestamp": 4.0},
    ]
    violations_b = [
        {"type": "no_seatbelt", "track_id": 3, "timestamp": 4.3},
    ]

    first = capture.capture(frame, violations_a)
    second = capture.capture(frame, violations_b)

    assert len(first) == 1
    assert second == []
