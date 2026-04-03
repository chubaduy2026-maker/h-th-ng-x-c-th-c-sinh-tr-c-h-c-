FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY tvu_face_attendance/requirements.txt ./tvu_face_attendance/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    python -m pip show opencv-python-headless && \
    python -c "import cv2; print('cv2', cv2.__version__)"

COPY . .

CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "10000"]
