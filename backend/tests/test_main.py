from fastapi.testclient import TestClient

import main
from schemas import AnalysisResponse, AnalysisSummary, DebugAnalysisResponse


client = TestClient(main.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_rejects_unsupported_extension():
    response = client.post(
        "/upload",
        files={"file": ("evidence.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type"


def test_upload_image_path_invokes_image_processor(monkeypatch):
    expected = AnalysisResponse(
        file_name="frame.jpg",
        is_video=False,
        duration_seconds=None,
        violations=[],
        summary=AnalysisSummary(total_violations=0, unique_types=0, avg_confidence=0),
    )

    def fake_process_image(_path: str, _file_name: str, include_rule_engine: bool = False):
        assert include_rule_engine is False
        return expected

    monkeypatch.setattr(main, "process_image", fake_process_image)

    response = client.post(
        "/upload",
        files={"file": ("frame.jpg", b"fake-binary", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["file_name"] == "frame.jpg"
    assert response.json()["is_video"] is False


def test_upload_debug_rejects_unsupported_extension():
    response = client.post(
        "/upload/debug",
        files={"file": ("evidence.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type"


def test_upload_debug_image_path_invokes_debug_processor(monkeypatch):
    expected = DebugAnalysisResponse(
        file_name="frame.jpg",
        is_video=False,
        duration_seconds=None,
        violations=[],
        summary=AnalysisSummary(total_violations=0, unique_types=0, avg_confidence=0),
        frame_detections=[],
    )

    def fake_process_image_debug(_path: str, _file_name: str, include_rule_engine: bool = False):
        assert include_rule_engine is False
        return expected

    monkeypatch.setattr(main, "process_image_debug", fake_process_image_debug)

    response = client.post(
        "/upload/debug",
        files={"file": ("frame.jpg", b"fake-binary", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["file_name"] == "frame.jpg"
    assert response.json()["is_video"] is False
    assert "frame_detections" in response.json()


def test_upload_image_path_can_enable_rule_engine(monkeypatch):
    expected = AnalysisResponse(
        file_name="frame.jpg",
        is_video=False,
        duration_seconds=None,
        violations=[],
        rule_engine_violations=[],
        summary=AnalysisSummary(total_violations=0, unique_types=0, avg_confidence=0),
    )

    def fake_process_image(_path: str, _file_name: str, include_rule_engine: bool = False):
        assert include_rule_engine is True
        return expected

    monkeypatch.setattr(main, "process_image", fake_process_image)

    response = client.post(
        "/upload?include_rule_engine=true",
        files={"file": ("frame.jpg", b"fake-binary", "image/jpeg")},
    )

    assert response.status_code == 200
    assert "rule_engine_violations" in response.json()
