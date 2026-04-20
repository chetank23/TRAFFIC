from pydantic import BaseModel, Field
from typing import Literal

ViolationType = Literal[
    "no_helmet",
    "no_seatbelt",
    "red_light",
    "wrong_lane",
    "mobile_usage",
    "overspeeding",
    "drunk_driving",
    "no_valid_license",
    "triple_riding",
    "no_parking",
    "dangerous_driving",
]


class BoundingBox(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


class Violation(BaseModel):
    id: str
    type: ViolationType
    label: str
    confidence: float = Field(ge=0, le=1)
    timestamp: float | None = Field(default=None, ge=0)
    box: BoundingBox
    vehicle_id: str | None = None
    description: str


class AnalysisSummary(BaseModel):
    total_violations: int
    unique_types: int
    avg_confidence: float = Field(ge=0, le=1)


class RuleEngineViolation(BaseModel):
    type: str
    bbox: tuple[float, float, float, float]
    confidence: float = Field(ge=0, le=1)
    timestamp: float | None = Field(default=None, ge=0)


class TrackedObject(BaseModel):
    id: int
    class_name: str = Field(serialization_alias="class")
    bbox: tuple[float, float, float, float]
    confidence: float = Field(ge=0, le=1)


class TrackingViolation(BaseModel):
    track_id: int
    type: str
    bbox: tuple[float, float, float, float]
    timestamp: float = Field(ge=0)


class AnalysisResponse(BaseModel):
    file_name: str
    is_video: bool
    duration_seconds: float | None = Field(default=None, ge=0)
    violations: list[Violation]
    rule_engine_violations: list[RuleEngineViolation] | None = None
    tracked_objects: list[TrackedObject] | None = None
    tracking_violations: list[TrackingViolation] | None = None
    summary: AnalysisSummary


class RawDetection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    box: BoundingBox


class FrameDetections(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    detections: list[RawDetection]
    tracked_objects: list[TrackedObject] | None = None
    tracking_violations: list[TrackingViolation] | None = None


class DebugAnalysisResponse(AnalysisResponse):
    frame_detections: list[FrameDetections]
