from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class EvidenceCapture:
    """Persist evidence images for violations with lightweight dedupe."""

    def __init__(
        self,
        root_dir: str | None = None,
        jpeg_quality: int | None = None,
        max_files: int | None = None,
    ) -> None:
        default_dir = Path(__file__).resolve().parent / "evidence"
        self.root_dir = Path(root_dir or os.environ.get("EVIDENCE_DIR") or str(default_dir))
        self.root_dir.mkdir(parents=True, exist_ok=True)

        env_quality = int(os.environ.get("EVIDENCE_JPEG_QUALITY", "75"))
        self.jpeg_quality = max(40, min(95, jpeg_quality if jpeg_quality is not None else env_quality))

        env_max_files = int(os.environ.get("EVIDENCE_MAX_FILES", "600"))
        self.max_files = max(20, max_files if max_files is not None else env_max_files)

        self.metadata_path = self.root_dir / "metadata.jsonl"
        self._seen_event_keys: set[str] = set()
        self._last_hash_by_key: dict[str, str] = {}

    def capture(self, frame: np.ndarray, violations: list[dict[str, Any]]) -> list[dict[str, str]]:
        if frame is None or frame.size == 0 or not violations:
            return []

        frame_hash = self._frame_hash(frame)
        captured: list[dict[str, str]] = []

        for item in violations:
            violation_type = self._normalize_type(item.get("type"))
            track_id = self._normalize_track_id(item.get("track_id"))
            timestamp = self._to_float_ts(item.get("timestamp"))

            rounded_ts = f"{timestamp:.3f}"
            event_key = f"{violation_type}:{track_id}:{rounded_ts}"
            dedupe_key = f"{violation_type}:{track_id}"

            if event_key in self._seen_event_keys:
                continue
            if self._last_hash_by_key.get(dedupe_key) == frame_hash:
                continue

            # '*' is invalid in Windows filenames, so '_' is used as a safe separator.
            safe_ts = rounded_ts.replace(".", "_")
            file_name = f"violation_{violation_type}_{track_id}_{safe_ts}.jpg"
            image_path = self.root_dir / file_name

            ok = cv2.imwrite(
                str(image_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
            )
            if not ok:
                continue

            metadata = {
                "type": violation_type,
                "time": self._format_iso_time(timestamp),
                "image_path": str(image_path),
            }

            self._append_metadata(metadata)
            self._seen_event_keys.add(event_key)
            self._last_hash_by_key[dedupe_key] = frame_hash
            captured.append(metadata)

        if captured:
            self._prune_old_files()
        return captured

    @staticmethod
    def _normalize_type(raw: Any) -> str:
        value = str(raw or "unknown").strip().lower()
        if value == "red_light_violation":
            return "red_light"
        return value

    @staticmethod
    def _normalize_track_id(raw: Any) -> str:
        try:
            return str(int(raw))
        except (TypeError, ValueError):
            return "na"

    @staticmethod
    def _to_float_ts(raw: Any) -> float:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _format_iso_time(timestamp: float) -> str:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.isoformat()

    @staticmethod
    def _frame_hash(frame: np.ndarray) -> str:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tiny = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        return hashlib.sha1(tiny.tobytes()).hexdigest()

    def _append_metadata(self, metadata: dict[str, str]) -> None:
        with self.metadata_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metadata, ensure_ascii=True) + "\n")

    def _prune_old_files(self) -> None:
        files = sorted(self.root_dir.glob("violation_*.jpg"), key=lambda p: p.stat().st_mtime)
        overflow = len(files) - self.max_files
        if overflow <= 0:
            return

        for old in files[:overflow]:
            try:
                old.unlink(missing_ok=True)
            except OSError:
                continue
