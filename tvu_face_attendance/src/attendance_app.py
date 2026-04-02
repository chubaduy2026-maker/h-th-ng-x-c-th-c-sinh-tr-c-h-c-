from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

try:
    from src.config import (
        CAMERA_INDEX,
        DEBOUNCE_SECONDS,
        DETECTION_SCORE_MIN,
        MIN_BRIGHTNESS,
        MIN_FACE_SIZE,
        MIN_LAPLACIAN_VAR,
        RECOGNITION_CONFIRM_FRAMES,
        THRESHOLD,
    )
    from src.database import pull_data, sync_attendance
    from src.face_engine import get_embedding_from_face, get_faces
    from src.matcher import FaceMatcher
except ImportError:
    from config import (
        CAMERA_INDEX,
        DEBOUNCE_SECONDS,
        DETECTION_SCORE_MIN,
        MIN_BRIGHTNESS,
        MIN_FACE_SIZE,
        MIN_LAPLACIAN_VAR,
        RECOGNITION_CONFIRM_FRAMES,
        THRESHOLD,
    )
    from database import pull_data, sync_attendance
    from face_engine import get_embedding_from_face, get_faces
    from matcher import FaceMatcher


BASE_DIR = Path(__file__).resolve().parent.parent


def _setup_logger(session_id: str) -> tuple[logging.Logger, Path]:
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"attendance_{session_id}.log"
    logger = logging.getLogger(f"attendance.{session_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger, log_path


def _export_csv_report(session_id: str, local_attendance: dict[str, dict[str, Any]]) -> Path:
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / f"attendance_{session_id}.csv"
    exported_at = datetime.now().isoformat(timespec="seconds")

    with report_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["session_id", "exported_at", "mssv", "name", "is_present", "status", "marked_at"]
        )

        for mssv in sorted(local_attendance.keys()):
            entry = local_attendance[mssv]
            is_present = bool(entry.get("is_present", False))
            writer.writerow(
                [
                    session_id,
                    exported_at,
                    mssv,
                    str(entry.get("name", "")),
                    int(is_present),
                    "Present" if is_present else "Absent",
                    str(entry.get("marked_at", "")),
                ]
            )

    return report_path


def _draw_box(frame: np.ndarray, box: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def _clip_box(box: np.ndarray, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def _face_quality_ok(frame: np.ndarray, box: np.ndarray) -> tuple[bool, str]:
    x1, y1, x2, y2 = _clip_box(box, frame.shape)
    width = x2 - x1
    height = y2 - y1

    if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
        return False, "TooFar"

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, "InvalidROI"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < MIN_BRIGHTNESS:
        return False, "LowLight"

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < MIN_LAPLACIAN_VAR:
        return False, "Blur"

    return True, "OK"


def run_attendance_session(logger: logging.Logger) -> dict[str, dict[str, Any]]:
    students = pull_data()
    if not students:
        raise RuntimeError("No student data available from cloud. Please register students first.")

    matcher = FaceMatcher(dimension=512)
    matcher.load_vectors(students)

    local_attendance: dict[str, dict[str, Any]] = {
        student["mssv"]: {
            "name": student.get("name", ""),
            "is_present": bool(student.get("is_present", False)),
        }
        for student in students
    }

    last_seen_time: dict[str, float] = {}
    hit_counter: dict[str, int] = {}

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam (device {CAMERA_INDEX}).")

    print("Attendance running...")
    print("Press 'q' to stop session.")
    print(
        "Quality gates: "
        f"det>={DETECTION_SCORE_MIN}, face>={MIN_FACE_SIZE}px, "
        f"brightness>={MIN_BRIGHTNESS}, blurVar>={MIN_LAPLACIAN_VAR}"
    )
    logger.info("Attendance session started with %s students.", len(students))

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Cannot read frame from webcam.")
                logger.warning("Cannot read frame from webcam.")
                continue

            try:
                faces = get_faces(frame)
            except Exception as exc:
                logger.exception("Face detection error: %s", exc)
                faces = []

            now = time.time()

            for face in faces:
                bbox = np.asarray(face.bbox, dtype=np.int32)
                det_score = float(getattr(face, "det_score", 1.0))

                if det_score < DETECTION_SCORE_MIN:
                    _draw_box(frame, bbox, f"LowConf({det_score:.2f})", (0, 165, 255))
                    continue

                quality_ok, reason = _face_quality_ok(frame, bbox)
                if not quality_ok:
                    _draw_box(frame, bbox, reason, (0, 165, 255))
                    continue

                try:
                    embedding = get_embedding_from_face(face)
                except Exception as exc:
                    logger.warning("Embedding extraction failed: %s", exc)
                    _draw_box(frame, bbox, "EmbErr", (0, 0, 255))
                    continue

                mssv = matcher.search(embedding, threshold=THRESHOLD)

                if mssv is None:
                    _draw_box(frame, bbox, "Unknown", (0, 0, 255))
                    continue

                name = matcher.get_name(mssv) or mssv
                hit_counter[mssv] = hit_counter.get(mssv, 0) + 1

                if hit_counter[mssv] < RECOGNITION_CONFIRM_FRAMES:
                    _draw_box(
                        frame,
                        bbox,
                        f"{name} (verify {hit_counter[mssv]}/{RECOGNITION_CONFIRM_FRAMES})",
                        (0, 255, 255),
                    )
                    continue

                last_time = last_seen_time.get(mssv, 0.0)
                is_debounced = (now - last_time) < DEBOUNCE_SECONDS

                if not is_debounced:
                    last_seen_time[mssv] = now
                    marked_at = datetime.now().isoformat(timespec="seconds")
                    if mssv in local_attendance:
                        local_attendance[mssv]["is_present"] = True
                        local_attendance[mssv]["marked_at"] = marked_at
                    else:
                        local_attendance[mssv] = {
                            "name": name,
                            "is_present": True,
                            "marked_at": marked_at,
                        }
                    print(f"[PRESENT] {mssv} - {name}")
                    logger.info("PRESENT | mssv=%s | name=%s", mssv, name)

                if is_debounced:
                    remain = max(0.0, DEBOUNCE_SECONDS - (now - last_time))
                    label = f"{name} ({remain:.1f}s)"
                else:
                    label = name

                _draw_box(frame, bbox, label, (0, 255, 0))

            present_count = sum(1 for item in local_attendance.values() if item["is_present"])
            total_count = len(local_attendance)
            status_text = f"Present: {present_count}/{total_count} | Threshold: {THRESHOLD:.2f}"
            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Press q to quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"ConfirmFrames: {RECOGNITION_CONFIRM_FRAMES} | Debounce: {DEBOUNCE_SECONDS:.1f}s",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Exam Attendance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Stop requested by user (q).")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Camera resources released.")

    return local_attendance


def main() -> None:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = _setup_logger(session_id)
    logger.info("Session id: %s", session_id)

    local_attendance = run_attendance_session(logger)
    present_count = sum(1 for item in local_attendance.values() if item["is_present"])
    print(f"Session ended. Local present count: {present_count}/{len(local_attendance)}")
    logger.info("Session ended. Local present count: %s/%s", present_count, len(local_attendance))

    try:
        sync_result = sync_attendance(local_attendance)
        print(
            "Sync completed: "
            f"matched={sync_result['matched']} | modified={sync_result['modified']}"
        )
        logger.info(
            "Sync completed. matched=%s | modified=%s",
            sync_result["matched"],
            sync_result["modified"],
        )
    except Exception as exc:
        print(f"Sync failed: {exc}")
        logger.exception("Sync failed: %s", exc)

    try:
        report_path = _export_csv_report(session_id, local_attendance)
        print(f"CSV report saved: {report_path}")
        logger.info("CSV report saved: %s", report_path)
    except Exception as exc:
        print(f"CSV export failed: {exc}")
        logger.exception("CSV export failed: %s", exc)

    print(f"Session log saved: {log_path}")


if __name__ == "__main__":
    main()
