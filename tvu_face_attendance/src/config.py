from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(str(BASE_DIR / ".env"), override=True, encoding="utf-8")

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = "TVU_Exam"
ATTENDANCE_LOG_COLLECTION = "attendance_logs"

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
SCAN_DURATION_SECONDS = max(1, int(os.getenv("SCAN_DURATION_SECONDS", "6")))
FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", "640"))

COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.40"))
LIVENESS_MIN_BRIGHTNESS = float(os.getenv("LIVENESS_MIN_BRIGHTNESS", "45"))
LIVENESS_MIN_LAPLACIAN_VAR = float(os.getenv("LIVENESS_MIN_LAPLACIAN_VAR", "70"))
LIVENESS_MIN_FACE_SIZE = int(os.getenv("LIVENESS_MIN_FACE_SIZE", "90"))


def validate_config() -> None:
    if not MONGO_URI or MONGO_URI == "your_connection_string_here":
        raise ValueError(
            "MONGO_URI is not configured. Please update the .env file with your MongoDB Atlas URI."
        )
