from __future__ import annotations

import base64
import binascii
import importlib.util
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

def _import_cv2_with_optional_bootstrap() -> tuple[Any | None, str]:
    try:
        import cv2 as _cv2  # type: ignore

        return _cv2, ""
    except Exception as exc:
        initial_error = f"{type(exc).__name__}: {exc}"

    auto_install = os.getenv("AUTO_INSTALL_CV2", "1" if os.getenv("RENDER") else "0") == "1"
    if not auto_install:
        return None, initial_error

    package_name = os.getenv("CV2_PIP_PACKAGE", "opencv-python-headless==4.10.0.84")
    try:
        # Render can occasionally skip/override build deps in misconfigured services.
        # This fallback attempts to recover cv2 at boot so the app can still run.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", package_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=240,
        )
        import cv2 as _cv2_after_install  # type: ignore

        return _cv2_after_install, ""
    except Exception as install_exc:
        install_error = f"{type(install_exc).__name__}: {install_exc}"
        return None, f"{initial_error}; auto-install failed ({package_name}) -> {install_error}"


cv2, _CV2_IMPORT_ERROR = _import_cv2_with_optional_bootstrap()

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

try:
    from src.config import (
        BASE_DIR,
        DETECTION_SCORE_MIN,
        MIN_FACE_SIZE,
        SCAN_DURATION_SECONDS,
        THRESHOLD,
    )
    from src.database import (
        add_attendance_log,
        add_student,
        delete_attendance_log,
        delete_student,
        delete_unknown_attendance_logs,
        get_all_students,
        get_attendance_logs,
        pull_data,
        update_attendance,
    )
except ImportError:
    from config import (
        BASE_DIR,
        DETECTION_SCORE_MIN,
        MIN_FACE_SIZE,
        SCAN_DURATION_SECONDS,
        THRESHOLD,
    )
    from database import (
        add_attendance_log,
        add_student,
        delete_attendance_log,
        delete_student,
        delete_unknown_attendance_logs,
        get_all_students,
        get_attendance_logs,
        pull_data,
        update_attendance,
    )
FACE_RUNTIME_AVAILABLE = cv2 is not None
FACE_RUNTIME_REASON = _CV2_IMPORT_ERROR
_VERCEL_LITE_MODE = os.getenv("VERCEL") == "1"

if FACE_RUNTIME_AVAILABLE:
    try:
        from src.face_engine import get_embedding_from_face, get_faces
        from src.matcher import FaceMatcher
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", "") in {
            "src",
            "src.face_engine",
            "src.matcher",
        }:
            try:
                from face_engine import get_embedding_from_face, get_faces
                from matcher import FaceMatcher
            except Exception as inner_exc:
                FACE_RUNTIME_AVAILABLE = False
                FACE_RUNTIME_REASON = f"{type(inner_exc).__name__}: {inner_exc}"
                get_embedding_from_face = None
                get_faces = None
                FaceMatcher = None
        else:
            FACE_RUNTIME_AVAILABLE = False
            FACE_RUNTIME_REASON = f"{type(exc).__name__}: {exc}"
            get_embedding_from_face = None
            get_faces = None
            FaceMatcher = None
else:
    get_embedding_from_face = None
    get_faces = None
    FaceMatcher = None


class _IndexStub:
    def __init__(self) -> None:
        self.ntotal = 0


class _DisabledFaceMatcher:
    def __init__(self) -> None:
        self.index = _IndexStub()

    def load_vectors(self, _students: list[dict[str, Any]]) -> None:
        return None

    def search(self, _vector: np.ndarray, threshold: float = 0.0) -> str:
        return ""

    def get_name(self, _mssv: str) -> str:
        return ""


def _face_runtime_message() -> str:
    if _VERCEL_LITE_MODE:
        return (
            "Face recognition is unavailable on this deployment. "
            "Vercel lite mode is enabled because AI dependencies exceed Vercel runtime limits."
        )

    base = "Face recognition is unavailable on this deployment."
    if FACE_RUNTIME_REASON:
        return f"{base} {FACE_RUNTIME_REASON}"
    return base


def _face_runtime_reason() -> str:
    if _VERCEL_LITE_MODE:
        return "Vercel lite mode: heavy AI dependencies are disabled for this deployment."
    if FACE_RUNTIME_AVAILABLE and cv2 is not None:
        return ""
    return FACE_RUNTIME_REASON or "Required AI runtime dependencies are missing."


def _ensure_face_runtime() -> None:
    if not FACE_RUNTIME_AVAILABLE or cv2 is None:
        raise HTTPException(status_code=503, detail=_face_runtime_message())


app = FastAPI(title="TVU Face Attendance Web")
templates = Jinja2Templates(directory=str(Path(BASE_DIR) / "templates"))

_attendance_lock = threading.Lock()
_attendance_matcher = (
    FaceMatcher(dimension=512)
    if FACE_RUNTIME_AVAILABLE and FaceMatcher is not None
    else _DisabledFaceMatcher()
)
_attendance_last_reload_ts = 0.0
_recent_attendance: list[dict[str, Any]] = []

_active_scan_session: dict[str, Any] | None = None
_SCAN_DURATION_SECONDS = max(5, min(6, int(SCAN_DURATION_SECONDS)))


def _decode_base64_image(image_base64: str) -> np.ndarray:
    if cv2 is None:
        raise HTTPException(status_code=503, detail=_face_runtime_message())

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


def _reload_attendance_matcher(force: bool = False) -> int:
    if not FACE_RUNTIME_AVAILABLE:
        return 0

    global _attendance_last_reload_ts

    now = time.time()
    with _attendance_lock:
        if not force and (now - _attendance_last_reload_ts) < 10.0:
            return int(_attendance_matcher.index.ntotal)

        students = pull_data()
        _attendance_matcher.load_vectors(students)
        _attendance_last_reload_ts = now
        return int(_attendance_matcher.index.ntotal)


def _append_recent_attendance(event: dict[str, Any]) -> None:
    with _attendance_lock:
        _recent_attendance.insert(0, event)
        if len(_recent_attendance) > 100:
            _recent_attendance.pop()


def _session_remaining_seconds(session: dict[str, Any]) -> int:
    elapsed = max(0.0, time.time() - float(session["started_at"]))
    return max(0, int(np.ceil(float(session["duration"]) - elapsed)))


def _pick_best_candidate(frame: np.ndarray) -> dict[str, Any] | None:
    if not FACE_RUNTIME_AVAILABLE:
        return None

    faces = get_faces(frame)
    best: dict[str, Any] | None = None

    for face in faces:
        det_score = float(getattr(face, "det_score", 1.0))
        bbox = np.asarray(face.bbox, dtype=np.int32)
        width = int(bbox[2] - bbox[0])
        height = int(bbox[3] - bbox[1])

        if det_score < DETECTION_SCORE_MIN:
            continue
        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            continue

        try:
            embedding = get_embedding_from_face(face)
            mssv = _attendance_matcher.search(embedding, threshold=THRESHOLD)
        except Exception:
            continue

        if not mssv:
            continue

        candidate_score = det_score
        candidate = {
            "mssv": mssv,
            "name": _attendance_matcher.get_name(mssv) or mssv,
            "det_score": det_score,
            "score": candidate_score,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def _finalize_scan_session(session: dict[str, Any]) -> dict[str, Any]:
    best_candidate = session.get("best_candidate")
    if not best_candidate:
        return {
            "status": "completed",
            "matched": False,
            "message": "Het 6 giay, khong tim thay sinh vien phu hop.",
            "student": None,
        }

    updated = update_attendance(best_candidate["mssv"])
    timestamp = datetime.now().astimezone()
    add_attendance_log(updated["mssv"], updated["name"])

    marked_at = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    event = {
        "mssv": updated["mssv"],
        "name": updated["name"],
        "marked_at": marked_at,
    }
    _append_recent_attendance(event)

    return {
        "status": "completed",
        "matched": True,
        "message": f"Da nhan dien: {updated['name']} ({updated['mssv']}).",
        "student": event,
    }


def _admin_log_rows() -> list[dict[str, str]]:
    def _to_local_text(ts: Any) -> tuple[str, str]:
        if not isinstance(ts, datetime):
            return "-", "-"

        # PyMongo may return naive UTC datetimes when tz_aware is disabled.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        local_time = ts.astimezone()
        return local_time.strftime("%H:%M:%S"), local_time.strftime("%d/%m/%Y")

    rows: list[dict[str, str]] = []
    for item in get_attendance_logs(limit=500):
        time_text, date_text = _to_local_text(item.get("timestamp"))

        rows.append(
            {
                "id": str(item.get("id", "")),
                "mssv": str(item.get("mssv", "")).strip(),
                "name": str(item.get("name", "")).strip(),
                "time_text": time_text,
                "date_text": date_text,
            }
        )
    return rows


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/attendance", status_code=307)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={
            "page_title": "Register",
            "face_runtime_available": FACE_RUNTIME_AVAILABLE and cv2 is not None,
            "face_runtime_reason": _face_runtime_reason(),
        },
    )


@app.get("/attendance", response_class=HTMLResponse)
async def attendance_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="attendance.html",
        context={
            "page_title": "Attendance",
            "face_runtime_available": FACE_RUNTIME_AVAILABLE and cv2 is not None,
            "face_runtime_reason": _face_runtime_reason(),
        },
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request) -> HTMLResponse:
    students: list[dict[str, Any]] = []
    log_rows: list[dict[str, str]] = []
    db_error = ""

    try:
        raw_students = get_all_students()
        students = []
        for item in raw_students:
            last_attended = item.get("last_attended_at")
            if isinstance(last_attended, datetime):
                if last_attended.tzinfo is None:
                    last_attended = last_attended.replace(tzinfo=timezone.utc)
                last_attended_text = last_attended.astimezone().strftime("%H:%M:%S %d/%m/%Y")
            else:
                last_attended_text = "-"

            new_item = dict(item)
            new_item["last_attended_text"] = last_attended_text
            students.append(new_item)

        log_rows = _admin_log_rows()
    except Exception as exc:
        db_error = str(exc)

    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={
            "page_title": "Admin",
            "students": students,
            "student_count": len(students),
            "attendance_logs": log_rows,
            "log_count": len(log_rows),
            "db_error": db_error,
        },
    )


@app.post("/api/register")
async def api_register(request: Request) -> dict[str, Any]:
    _ensure_face_runtime()

    payload = await request.json()

    mssv = str(payload.get("mssv", "")).strip()
    name = str(payload.get("name", "")).strip()
    image_base64 = str(payload.get("image_base64", "")).strip()

    if not mssv or not name or not image_base64:
        raise HTTPException(status_code=400, detail="mssv, name and image_base64 are required.")

    frame = _decode_base64_image(image_base64)
    faces = get_faces(frame)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    best_face = max(
        faces,
        key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
    )

    try:
        embedding = get_embedding_from_face(best_face)
        student = add_student(mssv=mssv, name=name, face_vector=embedding)
        _reload_attendance_matcher(force=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Register failed: {exc}") from exc

    return {"status": "ok", "student": student}


@app.delete("/api/admin/students/{mssv}")
async def api_delete_student(mssv: str) -> dict[str, Any]:
    try:
        result = delete_student(mssv)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc

    if result["deleted"] == 0:
        raise HTTPException(status_code=404, detail=f"Student not found: {result['mssv']}")

    _reload_attendance_matcher(force=True)
    return {"status": "ok", "deleted_mssv": result["mssv"]}


@app.delete("/api/admin/attendance-logs/cleanup-unknown")
async def api_cleanup_unknown_logs() -> dict[str, Any]:
    try:
        result = delete_unknown_attendance_logs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cleanup logs failed: {exc}") from exc

    return {
        "status": "ok",
        "deleted_count": result["deleted"],
    }


@app.delete("/api/admin/attendance-logs/{log_id}")
async def api_delete_attendance_log(log_id: str) -> dict[str, Any]:
    try:
        result = delete_attendance_log(log_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete log failed: {exc}") from exc

    if result["deleted"] == 0:
        raise HTTPException(status_code=404, detail="Log not found.")

    return {
        "status": "ok",
        "deleted_log_id": result["log_id"],
    }


@app.post("/api/attendance/reload")
async def api_attendance_reload() -> dict[str, Any]:
    _ensure_face_runtime()

    try:
        loaded = _reload_attendance_matcher(force=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc

    return {"status": "ok", "loaded": loaded}


@app.get("/api/attendance/recent")
async def api_recent_attendance() -> dict[str, Any]:
    with _attendance_lock:
        return {"recent": list(_recent_attendance[:20])}


@app.post("/api/attendance/session/start")
async def api_start_scan_session() -> dict[str, Any]:
    _ensure_face_runtime()

    global _active_scan_session

    loaded = _reload_attendance_matcher(force=False)
    if loaded <= 0:
        raise HTTPException(status_code=400, detail="No registered students. Please register first.")

    with _attendance_lock:
        if _active_scan_session is not None and _session_remaining_seconds(_active_scan_session) > 0:
            raise HTTPException(status_code=409, detail="Another scan session is running.")

        _active_scan_session = {
            "session_id": uuid4().hex,
            "started_at": time.time(),
            "duration": _SCAN_DURATION_SECONDS,
            "best_candidate": None,
        }

        return {
            "status": "ok",
            "session_id": _active_scan_session["session_id"],
            "duration_seconds": _SCAN_DURATION_SECONDS,
        }


@app.post("/api/attendance/session/frame")
async def api_scan_session_frame(request: Request) -> dict[str, Any]:
    _ensure_face_runtime()

    global _active_scan_session

    payload = await request.json()
    session_id = str(payload.get("session_id", "")).strip()
    image_base64 = str(payload.get("image_base64", "")).strip()

    if not session_id or not image_base64:
        raise HTTPException(status_code=400, detail="session_id and image_base64 are required.")

    with _attendance_lock:
        session = _active_scan_session

    if session is None or session.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Scan session not found.")

    frame = _decode_base64_image(image_base64)
    candidate = _pick_best_candidate(frame)

    session_to_finalize: dict[str, Any] | None = None

    with _attendance_lock:
        session = _active_scan_session
        if session is None or session.get("session_id") != session_id:
            raise HTTPException(status_code=404, detail="Scan session not found.")

        if candidate is not None:
            best = session.get("best_candidate")
            if best is None or float(candidate["score"]) > float(best.get("score", -1.0)):
                session["best_candidate"] = candidate

        remaining_seconds = _session_remaining_seconds(session)
        preview = session.get("best_candidate")

        if remaining_seconds > 0:
            return {
                "status": "running",
                "remaining_seconds": remaining_seconds,
                "preview": {
                    "mssv": preview.get("mssv"),
                    "name": preview.get("name"),
                } if preview else None,
            }

        session_to_finalize = dict(session)
        _active_scan_session = None

    if session_to_finalize is None:
        raise HTTPException(status_code=500, detail="Internal scan finalize error.")

    result = _finalize_scan_session(session_to_finalize)
    return {
        **result,
        "remaining_seconds": 0,
    }


@app.post("/api/attendance/session/cancel")
async def api_cancel_scan_session(request: Request) -> dict[str, Any]:
    global _active_scan_session

    payload = await request.json()
    session_id = str(payload.get("session_id", "")).strip()

    with _attendance_lock:
        if _active_scan_session is None:
            return {"status": "ok", "message": "No active session."}

        if session_id and _active_scan_session.get("session_id") != session_id:
            raise HTTPException(status_code=404, detail="Scan session not found.")

        _active_scan_session = None

    return {"status": "ok", "message": "Session canceled."}


@app.post("/api/attendance/session/finish")
async def api_finish_scan_session(request: Request) -> dict[str, Any]:
    global _active_scan_session

    payload = await request.json()
    session_id = str(payload.get("session_id", "")).strip()

    with _attendance_lock:
        if _active_scan_session is None:
            return {
                "status": "completed",
                "matched": False,
                "message": "No active session.",
                "student": None,
                "remaining_seconds": 0,
            }

        if session_id and _active_scan_session.get("session_id") != session_id:
            raise HTTPException(status_code=404, detail="Scan session not found.")

        session = _active_scan_session
        _active_scan_session = None

    result = _finalize_scan_session(session)
    return {
        **result,
        "remaining_seconds": 0,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "face_runtime_available": FACE_RUNTIME_AVAILABLE and cv2 is not None,
        "face_runtime_reason": _face_runtime_reason(),
        "cv2_version": getattr(cv2, "__version__", ""),
        "cv2_module_found": importlib.util.find_spec("cv2") is not None,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "render_git_commit": os.getenv("RENDER_GIT_COMMIT", ""),
        "release": os.getenv("RELEASE", "2026-04-03-render-cv2-fix"),
    }
