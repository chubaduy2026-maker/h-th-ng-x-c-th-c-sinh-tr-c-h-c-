from __future__ import annotations

import base64
import binascii
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient


def _import_process_frame_func() -> Any:
    try:
        from src.face_engine import process_frame as imported_process_frame

        return imported_process_frame
    except Exception as src_exc:
        try:
            from face_engine import process_frame as imported_process_frame

            return imported_process_frame
        except Exception as local_exc:
            raise local_exc from src_exc


def _install_face_runtime_packages() -> str:
    packages_env = os.getenv("FACE_RUNTIME_PIP_PACKAGES", "insightface onnxruntime")
    packages = [item.strip() for item in packages_env.split() if item.strip()]
    if not packages:
        return "FACE_RUNTIME_PIP_PACKAGES is empty."

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", *packages],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
        )
        return ""
    except Exception as exc:
        return f"Auto-install failed: {type(exc).__name__}: {exc}"

try:
    from src.config import (
        ATTENDANCE_LOG_COLLECTION,
        BASE_DIR,
        DB_NAME,
        MONGO_URI,
        SCAN_DURATION_SECONDS,
        validate_config,
    )
except ImportError:
    from config import (
        ATTENDANCE_LOG_COLLECTION,
        BASE_DIR,
        DB_NAME,
        MONGO_URI,
        SCAN_DURATION_SECONDS,
        validate_config,
    )

process_frame = None
FACE_RUNTIME_AVAILABLE = cv2 is not None
FACE_RUNTIME_REASON = "" if cv2 is not None else "OpenCV is unavailable."

if FACE_RUNTIME_AVAILABLE:
    try:
        process_frame = _import_process_frame_func()
    except Exception as exc:
        auto_install = os.getenv("AUTO_INSTALL_FACE_RUNTIME", "1" if os.getenv("RENDER") else "0") == "1"
        if auto_install:
            install_error = _install_face_runtime_packages()
            if not install_error:
                try:
                    process_frame = _import_process_frame_func()
                except Exception as retry_exc:
                    FACE_RUNTIME_AVAILABLE = False
                    FACE_RUNTIME_REASON = f"{type(retry_exc).__name__}: {retry_exc}"
            else:
                FACE_RUNTIME_AVAILABLE = False
                FACE_RUNTIME_REASON = install_error
        else:
            FACE_RUNTIME_AVAILABLE = False
            FACE_RUNTIME_REASON = f"{type(exc).__name__}: {exc}"

if process_frame is None:
    FACE_RUNTIME_AVAILABLE = False
    if not FACE_RUNTIME_REASON:
        FACE_RUNTIME_REASON = "Face engine is unavailable."


class ScanState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.is_scanning = False
        self.start_time = 0.0
        self.duration = int(SCAN_DURATION_SECONDS)
        self.best_candidate: dict[str, Any] | None = None
        self.last_status = "Trang thai cho"
        self.last_result: dict[str, Any] | None = None
        self.last_error = ""

    def start(self) -> dict[str, Any]:
        with self.lock:
            if self.is_scanning:
                elapsed = time.time() - self.start_time
                remaining = max(0, int(np.ceil(self.duration - elapsed)))
                return {
                    "status": "busy",
                    "message": "He thong dang quet.",
                    "remaining_seconds": remaining,
                }

            self.is_scanning = True
            self.start_time = time.time()
            self.best_candidate = None
            self.last_status = "Dang quet..."
            self.last_result = None
            self.last_error = ""

            return {
                "status": "ok",
                "message": f"Da bat dau phien quet {self.duration} giay.",
                "remaining_seconds": self.duration,
            }


app = FastAPI(title="TVU Face Attendance")
templates = Jinja2Templates(directory=str(Path(BASE_DIR) / "templates"))
_state = ScanState()

_db_client: MongoClient | None = None
_attendance_logs = None
_db_error = ""

try:
    validate_config()
    _db_client = MongoClient(MONGO_URI)
    _attendance_logs = _db_client[DB_NAME][ATTENDANCE_LOG_COLLECTION]
except Exception as exc:
    _db_error = f"Database unavailable: {exc}"


def _insert_attendance_log(mssv: str, name: str) -> bool:
    if _attendance_logs is None:
        return False

    _attendance_logs.insert_one(
        {
            "mssv": mssv,
            "name": name,
            "timestamp": datetime.now(timezone.utc),
        }
    )
    return True


def _remaining_seconds() -> int:
    elapsed = max(0.0, time.time() - _state.start_time)
    return max(0, int(np.ceil(_state.duration - elapsed)))


def _decode_base64_image(image_base64: str) -> np.ndarray:
    if cv2 is None:
        raise HTTPException(status_code=503, detail="OpenCV runtime is unavailable.")

    payload = image_base64.strip()
    if payload.startswith("data:image") and "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image.") from exc

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image data.")

    return frame


def _update_best_candidate(result: dict[str, Any]) -> None:
    if not result.get("face_found"):
        return

    bbox = result.get("bbox")
    mssv = result.get("mssv")
    name = result.get("name")
    if not bbox or not mssv or not name:
        return

    candidate_score = float(result.get("candidate_score", 0.0))
    det_score = float(result.get("det_confidence", 0.0))
    match_score = float(result.get("match_score", 0.0))

    if candidate_score <= 0:
        return

    current = _state.best_candidate
    if current is None or candidate_score > float(current.get("candidate_score", 0.0)):
        _state.best_candidate = {
            "bbox": bbox,
            "mssv": mssv,
            "name": name,
            "candidate_score": candidate_score,
            "det_confidence": det_score,
            "match_score": match_score,
        }


def _finalize_scan_once() -> None:
    candidate = _state.best_candidate
    _state.is_scanning = False

    if candidate is None:
        _state.last_status = "Het 6 giay: Khong tim thay sinh vien phu hop"
        _state.last_result = {
            "matched": False,
            "message": _state.last_status,
            "mssv": "",
            "name": "",
        }
        return

    inserted = _insert_attendance_log(candidate["mssv"], candidate["name"])
    if inserted:
        _state.last_status = f"Da nhan dien: {candidate['name']} ({candidate['mssv']})"
    else:
        _state.last_status = (
            f"Da nhan dien: {candidate['name']} ({candidate['mssv']}), "
            "nhung khong ghi duoc log MongoDB"
        )

    _state.last_result = {
        "matched": True,
        "message": _state.last_status,
        "mssv": candidate["mssv"],
        "name": candidate["name"],
    }


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "db_error": _db_error,
            "face_runtime_available": FACE_RUNTIME_AVAILABLE,
            "face_runtime_reason": FACE_RUNTIME_REASON,
        },
    )


@app.get("/attendance", include_in_schema=False)
async def attendance_alias() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request) -> HTMLResponse:
    rows: list[dict[str, str]] = []
    if _attendance_logs is not None:
        for idx, item in enumerate(_attendance_logs.find().sort("timestamp", -1).limit(500), start=1):
            ts = item.get("timestamp")
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ts_local = ts.astimezone()
                time_text = ts_local.strftime("%H:%M:%S")
                date_text = ts_local.strftime("%d/%m/%Y")
            else:
                time_text = "-"
                date_text = "-"

            rows.append(
                {
                    "stt": str(idx),
                    "mssv": str(item.get("mssv", "")),
                    "name": str(item.get("name", "")),
                    "time_text": time_text,
                    "date_text": date_text,
                }
            )

    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={"rows": rows, "db_error": _db_error},
    )


@app.post("/start_scan")
async def start_scan() -> dict[str, Any]:
    if not FACE_RUNTIME_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Face runtime unavailable on this deployment. "
                + (FACE_RUNTIME_REASON or "Missing AI dependencies.")
            ),
        )
    return _state.start()


@app.post("/process_frame")
async def process_scan_frame(request: Request) -> dict[str, Any]:
    if not FACE_RUNTIME_AVAILABLE or process_frame is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Face runtime unavailable on this deployment. "
                + (FACE_RUNTIME_REASON or "Missing AI dependencies.")
            ),
        )

    payload = await request.json()
    image_base64 = str(payload.get("image_base64", "")).strip()
    if not image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required.")

    with _state.lock:
        if not _state.is_scanning:
            return {
                "is_scanning": False,
                "remaining_seconds": 0,
                "status_text": _state.last_status,
                "last_result": _state.last_result,
                "last_error": _state.last_error,
                "preview": None,
            }

    frame = _decode_base64_image(image_base64)

    try:
        result = process_frame(frame)
    except Exception as exc:
        with _state.lock:
            _state.last_error = str(exc)
        result = {
            "face_found": False,
            "bbox": None,
            "mssv": None,
            "name": None,
            "det_confidence": 0.0,
            "match_score": 0.0,
            "candidate_score": 0.0,
        }

    preview = None
    with _state.lock:
        _update_best_candidate(result)

        if _remaining_seconds() <= 0 and _state.is_scanning:
            _finalize_scan_once()

        if _state.best_candidate is not None:
            preview = {
                "mssv": _state.best_candidate["mssv"],
                "name": _state.best_candidate["name"],
            }

        return {
            "is_scanning": _state.is_scanning,
            "remaining_seconds": _remaining_seconds() if _state.is_scanning else 0,
            "status_text": _state.last_status if not _state.is_scanning else "Dang quet...",
            "last_result": _state.last_result,
            "last_error": _state.last_error,
            "preview": preview,
        }


@app.get("/scan_status")
async def scan_status() -> dict[str, Any]:
    with _state.lock:
        if _state.is_scanning and _remaining_seconds() <= 0:
            _finalize_scan_once()

        if _state.is_scanning:
            return {
                "is_scanning": True,
                "remaining_seconds": _remaining_seconds(),
                "status_text": "Dang quet...",
                "last_result": _state.last_result,
                "last_error": _state.last_error,
            }

        return {
            "is_scanning": False,
            "remaining_seconds": 0,
            "status_text": _state.last_status,
            "last_result": _state.last_result,
            "last_error": _state.last_error,
        }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "db_ready": _attendance_logs is not None,
        "is_scanning": _state.is_scanning,
        "camera_mode": "browser",
        "face_runtime_available": FACE_RUNTIME_AVAILABLE,
        "face_runtime_reason": FACE_RUNTIME_REASON,
    }
