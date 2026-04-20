import cv2
import numpy as np

import utils
from detector import Detection


def test_process_image_returns_violation_payload(monkeypatch, tmp_path):
    image_path = tmp_path / "frame.jpg"
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    fake_detections = [
        Detection(2, "car", 0.92, (100, 180, 320, 420)),
        Detection(0, "person", 0.86, (160, 200, 250, 360)),
        Detection(9, "traffic light", 0.81, (20, 30, 80, 150)),
    ]

    monkeypatch.setattr(utils, "detect_objects", lambda _img: fake_detections)

    result = utils.process_image(str(image_path), "frame.jpg")

    assert result.is_video is False
    assert result.file_name == "frame.jpg"
    assert len(result.violations) > 0
    assert result.summary.total_violations == len(result.violations)


def test_process_video_returns_duration_and_timestamps(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"

    fps = 10.0
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480)
    )
    for i in range(20):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50 + i * 8, 220), (170 + i * 8, 340), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    def fake_detect_objects(frame):
        # Return realistic traffic detections used by the current rule set.
        nonlocal_x = int(np.argmax(frame[220, :, 0] > 0))
        return [
            Detection(2, "car", 0.9, (nonlocal_x, 220, min(nonlocal_x + 120, 639), 340)),
            Detection(9, "traffic light", 0.85, (18, 30, 90, 160)),
        ]

    monkeypatch.setattr(utils, "detect_objects", fake_detect_objects)

    result = utils.process_video(str(video_path), "clip.mp4")

    assert result.is_video is True
    assert result.duration_seconds is not None
    assert result.duration_seconds > 0
    assert isinstance(result.violations, list)
    if result.violations:
        assert result.violations[0].timestamp is not None
