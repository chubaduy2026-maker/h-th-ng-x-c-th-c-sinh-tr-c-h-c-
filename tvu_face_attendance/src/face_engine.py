from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

try:
    from src.config import (
        COSINE_THRESHOLD,
        FACE_DET_SIZE,
        LIVENESS_MIN_BRIGHTNESS,
        LIVENESS_MIN_FACE_SIZE,
        LIVENESS_MIN_LAPLACIAN_VAR,
    )
except ImportError:
    from config import (
        COSINE_THRESHOLD,
        FACE_DET_SIZE,
        LIVENESS_MIN_BRIGHTNESS,
        LIVENESS_MIN_FACE_SIZE,
        LIVENESS_MIN_LAPLACIAN_VAR,
    )


MODEL_NAME = "buffalo_l"
_DET_SIZE = (max(320, FACE_DET_SIZE), max(320, FACE_DET_SIZE))
_face_app: Optional[FaceAnalysis] = None


class FaceNotFoundError(ValueError):
    pass


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm <= 0:
        raise ValueError("Invalid embedding with zero norm.")
    return embedding / norm


def _get_face_app() -> FaceAnalysis:
    global _face_app

    if _face_app is not None:
        return _face_app

    app = FaceAnalysis(name=MODEL_NAME)
    try:
        app.prepare(ctx_id=0, det_size=_DET_SIZE)
    except Exception:
        app.prepare(ctx_id=-1, det_size=_DET_SIZE)

    _face_app = app
    return _face_app


def _build_sample_vectors() -> list[dict[str, Any]]:
    rng = np.random.default_rng(2026)
    vec1 = rng.normal(0.0, 1.0, 512).astype(np.float32)
    vec2 = rng.normal(0.0, 1.0, 512).astype(np.float32)

    vec1 = _normalize_embedding(vec1)
    vec2 = _normalize_embedding(vec2)

    return [
        {
            "mssv": "110122001",
            "name": "Nguyen Van A",
            "vector": vec1,
        },
        {
            "mssv": "110122002",
            "name": "Tran Thi B",
            "vector": vec2,
        },
    ]


_SAMPLE_STUDENTS = _build_sample_vectors()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _largest_face(faces: list[Any]) -> Any:
    return max(faces, key=lambda face: float(getattr(face, "det_score", 0.0)))


def _clip_box(box: np.ndarray, shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def _basic_liveness(frame: np.ndarray, box: np.ndarray) -> tuple[bool, str]:
    x1, y1, x2, y2 = _clip_box(box, frame.shape)
    face_w = max(0, x2 - x1)
    face_h = max(0, y2 - y1)

    if face_w < LIVENESS_MIN_FACE_SIZE or face_h < LIVENESS_MIN_FACE_SIZE:
        return False, "Face too small"

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, "Invalid ROI"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < LIVENESS_MIN_BRIGHTNESS:
        return False, "Low brightness"

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < LIVENESS_MIN_LAPLACIAN_VAR:
        return False, "Image blur too high"

    return True, "Live"


def process_frame(frame: np.ndarray) -> dict[str, Any]:
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError("frame must be a valid BGR image (numpy array).")

    faces = _get_face_app().get(frame)
    if not faces:
        return {
            "face_found": False,
            "bbox": None,
            "det_confidence": 0.0,
            "mssv": None,
            "name": None,
            "match_score": 0.0,
            "is_live": False,
            "liveness_warning": "No face",
            "candidate_score": 0.0,
        }

    best_face = _largest_face(faces)
    bbox = np.asarray(best_face.bbox, dtype=np.int32)
    det_confidence = float(getattr(best_face, "det_score", 0.0))
    embedding = _normalize_embedding(np.asarray(best_face.embedding, dtype=np.float32).flatten())

    best_match: dict[str, Any] | None = None
    for student in _SAMPLE_STUDENTS:
        score = _cosine_similarity(embedding, student["vector"])
        if best_match is None or score > best_match["score"]:
            best_match = {
                "mssv": student["mssv"],
                "name": student["name"],
                "score": float(score),
            }

    is_live, live_msg = _basic_liveness(frame, bbox)
    matched = bool(best_match and best_match["score"] >= COSINE_THRESHOLD and is_live)
    candidate_score = (
        0.55 * float(det_confidence) + 0.45 * float(best_match["score"])
        if best_match and matched
        else 0.0
    )

    return {
        "face_found": True,
        "bbox": [int(v) for v in bbox],
        "det_confidence": det_confidence,
        "mssv": best_match["mssv"] if matched and best_match else None,
        "name": best_match["name"] if matched and best_match else None,
        "match_score": float(best_match["score"]) if best_match else 0.0,
        "is_live": is_live,
        "liveness_warning": "" if is_live else live_msg,
        "candidate_score": float(candidate_score),
    }


def get_faces(img_array: np.ndarray) -> list[Any]:
    if img_array is None or not isinstance(img_array, np.ndarray) or img_array.ndim != 3:
        raise ValueError("img_array must be a valid BGR image.")
    return _get_face_app().get(img_array)


def get_embedding_from_face(face: Any) -> np.ndarray:
    embedding = np.asarray(face.embedding, dtype=np.float32).flatten()
    if embedding.shape[0] != 512:
        raise ValueError(f"Unexpected embedding size: {embedding.shape[0]}. Expected 512.")
    return _normalize_embedding(embedding)


def get_embedding(img_array: np.ndarray) -> np.ndarray:
    faces = get_faces(img_array)
    if not faces:
        raise FaceNotFoundError("No face detected in the input image.")
    return get_embedding_from_face(_largest_face(faces))
