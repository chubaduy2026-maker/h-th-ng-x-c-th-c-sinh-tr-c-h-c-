from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient

try:
    from src.config import (
        ATTENDANCE_LOG_COLLECTION,
        BASE_DIR,
        CAMERA_INDEX,
        DB_NAME,
        MONGO_URI,
        SCAN_DURATION_SECONDS,
        validate_config,
    )
    from src.face_engine import process_frame
except ImportError:
    from config import (
        ATTENDANCE_LOG_COLLECTION,
        BASE_DIR,
        CAMERA_INDEX,
        DB_NAME,
        MONGO_URI,
        SCAN_DURATION_SECONDS,
        validate_config,
    )
    from face_engine import process_frame


class CameraStream:
    def __init__(self, camera_index: int) -> None:
        self.camera_index = camera_index
        self._lock = threading.Lock()
        self._capture: cv2.VideoCapture | None = None

    def read(self) -> tuple[bool, np.ndarray]:
        with self._lock:
            if self._capture is None:
                self._capture = cv2.VideoCapture(self.camera_index)
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            ok, frame = self._capture.read()
            if ok and frame is not None:
                return True, frame

            fallback = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(
                fallback,
                "Camera unavailable",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return False, fallback


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
                "message": "Da bat dau phien quet 6 giay.",
                "remaining_seconds": self.duration,
            }


app = FastAPI(title="TVU Face Attendance")
templates = Jinja2Templates(directory=str(Path(BASE_DIR) / "templates"))

_camera = CameraStream(CAMERA_INDEX)
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


def _draw_overlay(frame: np.ndarray) -> np.ndarray:
    with _state.lock:
        is_scanning = _state.is_scanning
        remaining = _remaining_seconds() if is_scanning else 0
        status_text = _state.last_status

    if is_scanning:
        cv2.putText(
            frame,
            f"Scan: {remaining}s",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            status_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

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


def _frame_generator():
    while True:
        _, frame = _camera.read()

        with _state.lock:
            scanning = _state.is_scanning

        if scanning:
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

            with _state.lock:
                _update_best_candidate(result)

                bbox = result.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = "Scanning"
                    if result.get("name"):
                        label = str(result["name"])
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                if _remaining_seconds() <= 0 and _state.is_scanning:
                    _finalize_scan_once()

        frame = _draw_overlay(frame)
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        payload = encoded.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        )


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"db_error": _db_error},
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
    return _state.start()


@app.get("/scan_status")
async def scan_status() -> dict[str, Any]:
    with _state.lock:
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


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "db_ready": _attendance_logs is not None,
        "is_scanning": _state.is_scanning,
    }
