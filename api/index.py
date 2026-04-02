from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "tvu_face_attendance"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.web_app import app
