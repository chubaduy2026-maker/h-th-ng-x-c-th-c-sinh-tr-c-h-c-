from __future__ import annotations

import cv2
import numpy as np

try:
    from src.config import (
        CAMERA_INDEX,
        DETECTION_SCORE_MIN,
        MIN_BRIGHTNESS,
        MIN_FACE_SIZE,
        MIN_LAPLACIAN_VAR,
    )
    from src.database import register_student
    from src.face_engine import FaceNotFoundError, get_embedding_from_face, get_faces
except ImportError:
    from config import (
        CAMERA_INDEX,
        DETECTION_SCORE_MIN,
        MIN_BRIGHTNESS,
        MIN_FACE_SIZE,
        MIN_LAPLACIAN_VAR,
    )
    from database import register_student
    from face_engine import FaceNotFoundError, get_embedding_from_face, get_faces


def _prompt_student_info() -> tuple[str, str]:
    mssv = input("Enter MSSV: ").strip()
    name = input("Enter full name: ").strip()

    if not mssv:
        raise ValueError("MSSV cannot be empty.")
    if not name:
        raise ValueError("Name cannot be empty.")

    return mssv, name


def _select_best_face(faces: list[object]) -> object:
    if not faces:
        raise FaceNotFoundError("No face detected in current frame.")

    return max(
        faces,
        key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
    )


def _face_quality_ok(frame: np.ndarray, bbox: np.ndarray) -> tuple[bool, str]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    face_w = x2 - x1
    face_h = y2 - y1
    if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
        return False, "Face is too small. Move closer to camera."

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, "Invalid face region."

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < MIN_BRIGHTNESS:
        return False, "Lighting is too dark. Increase room light."

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < MIN_LAPLACIAN_VAR:
        return False, "Image is blurry. Keep your face steady."

    return True, "OK"


def main() -> None:
    print("=== Student Face Registration ===")
    print("Press 'c' to capture and register")
    print("Press 'n' to input another student")
    print("Press 'q' to quit")

    mssv, name = _prompt_student_info()

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam (device {CAMERA_INDEX}).")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Cannot read frame from webcam.")
                continue

            cv2.putText(
                frame,
                f"MSSV: {mssv} | Name: {name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Keys: c=Capture, n=New student, q=Quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Registration Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                try:
                    faces = get_faces(frame)
                    best_face = _select_best_face(faces)

                    det_score = float(getattr(best_face, "det_score", 1.0))
                    if det_score < DETECTION_SCORE_MIN:
                        print(
                            "Face detection confidence is low "
                            f"({det_score:.2f} < {DETECTION_SCORE_MIN:.2f}). Try again."
                        )
                        continue

                    quality_ok, reason = _face_quality_ok(
                        frame,
                        np.asarray(best_face.bbox, dtype=np.int32),
                    )
                    if not quality_ok:
                        print(f"Capture quality failed: {reason}")
                        continue

                    embedding = get_embedding_from_face(best_face)
                    result = register_student(mssv=mssv, name=name, vector=embedding)
                    print(
                        f"Registered: MSSV={result['mssv']} | Name={result['name']} | ID={result['id']}"
                    )
                except FaceNotFoundError:
                    print("No face detected. Please face the camera and try again.")
                except Exception as exc:
                    print(f"Registration failed: {exc}")

            elif key == ord("n"):
                try:
                    mssv, name = _prompt_student_info()
                except ValueError as exc:
                    print(f"Input error: {exc}")

            elif key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
