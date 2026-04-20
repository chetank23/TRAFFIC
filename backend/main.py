from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from schemas import AnalysisResponse, DebugAnalysisResponse
from utils import is_image_file, is_video_file, process_image, process_image_debug, process_video, process_video_debug


app = FastAPI(title="Sentry Traffic AI API", version="1.0.0")

default_origins = ",".join(
    [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:8081",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
    ]
)
allowed_origins = os.environ.get("ALLOWED_ORIGINS", default_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _validate_and_store_upload(file: UploadFile) -> tuple[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    suffix = Path(file.filename).suffix.lower()
    if not suffix:
        raise HTTPException(status_code=400, detail="File extension is required")

    if not (is_image_file(file.filename) or is_video_file(file.filename)):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    return temp_path, file.filename


@app.post("/upload", response_model=AnalysisResponse)
async def upload_media(
    file: UploadFile = File(...),
    include_rule_engine: bool = Query(default=False),
    include_tracking: bool = Query(default=False),
    include_violation_engine: bool = Query(default=False),
) -> AnalysisResponse:
    temp_path, original_name = await _validate_and_store_upload(file)

    try:
        if is_video_file(original_name):
            return process_video(
                temp_path,
                original_name,
                include_rule_engine=include_rule_engine,
                include_tracking=include_tracking,
                include_violation_engine=include_violation_engine,
            )
        return process_image(
            temp_path,
            original_name,
            include_rule_engine=include_rule_engine,
            include_tracking=include_tracking,
            include_violation_engine=include_violation_engine,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@app.post("/upload/debug", response_model=DebugAnalysisResponse)
async def upload_media_debug(
    file: UploadFile = File(...),
    include_rule_engine: bool = Query(default=False),
    include_tracking: bool = Query(default=False),
    include_violation_engine: bool = Query(default=False),
) -> DebugAnalysisResponse:
    temp_path, original_name = await _validate_and_store_upload(file)

    try:
        if is_video_file(original_name):
            return process_video_debug(
                temp_path,
                original_name,
                include_rule_engine=include_rule_engine,
                include_tracking=include_tracking,
                include_violation_engine=include_violation_engine,
            )
        return process_image_debug(
            temp_path,
            original_name,
            include_rule_engine=include_rule_engine,
            include_tracking=include_tracking,
            include_violation_engine=include_violation_engine,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
