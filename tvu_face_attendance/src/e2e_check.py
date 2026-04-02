from __future__ import annotations

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

try:
    from src.config import BASE_DIR, validate_config
    from src.database import pull_data
    from src.face_engine import FaceNotFoundError, get_embedding
    from src.matcher import FaceMatcher
except ImportError:
    from config import BASE_DIR, validate_config
    from database import pull_data
    from face_engine import FaceNotFoundError, get_embedding
    from matcher import FaceMatcher


def _print_step(name: str) -> None:
    print(f"\n=== {name} ===")


def main() -> None:
    failed = False

    try:
        _print_step("1) Validate configuration")
        validate_config()
        print("PASS: MONGO_URI is configured.")

        _print_step("2) Pull data from MongoDB")
        students = pull_data()
        print(f"PASS: Pulled {len(students)} student records.")

        _print_step("3) Verify FAISS matching self-test")
        if students:
            matcher = FaceMatcher(dimension=512)
            matcher.load_vectors(students)
            first = students[0]
            query = np.asarray(first["embedding"], dtype=np.float32)
            matched_mssv = matcher.search(query, threshold=0.0)
            if matched_mssv != first["mssv"]:
                raise RuntimeError(
                    "FAISS self-test failed: first vector did not match its own MSSV."
                )
            print(f"PASS: FAISS matched {matched_mssv} correctly.")
        else:
            print("SKIP: No students found, register data first to test FAISS behavior.")

        _print_step("4) Face model smoke test from data folder")
        data_dir = BASE_DIR / "data"
        image_paths = sorted(
            [
                p
                for p in data_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )

        if not image_paths:
            print("SKIP: No test image in data folder.")
        else:
            img_path = image_paths[0]
            img = cv2.imread(str(img_path))
            if img is None:
                raise RuntimeError(f"Cannot read image: {img_path}")

            try:
                emb = get_embedding(img)
                print(f"PASS: Extracted embedding from {img_path.name}, size={emb.shape[0]}.")
            except FaceNotFoundError:
                print(
                    "WARN: Test image found but no face detected. "
                    "Please place a clear frontal face image in data folder."
                )

    except Exception as exc:
        failed = True
        print(f"FAIL: {exc}")
        traceback.print_exc()

    _print_step("Result")
    if failed:
        print("E2E check failed.")
        sys.exit(1)

    print("E2E check finished successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
