"""
face_recog.py — Lightweight face enrollment & verification using OpenCV LBPH.

No external dlib / face_recognition package needed — only opencv-python (already installed).

Flow:
  1. verify_face page  → POST /api/face/enroll  → FaceRecognizer.enroll(username, frame)
  2. exam frame upload → vision_monitor calls    → FaceRecognizer.verify(username, frame)
"""

import os
import pickle
import numpy as np
import cv2

_DIR       = os.path.dirname(__file__)
MODELS_DIR = os.path.join(os.path.dirname(_DIR), "static", "face_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Haar cascade (bundled with OpenCV)
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

# LBPH confidence ≤ this = match (lower = more similar).
# LBPH distance is a squared L2 distance in histogram space:
#   0   = identical frames
#   50  = very likely same person (different day / slight lighting change)
#   120 = reasonable boundary — same person in varied conditions
#   200+ = likely different person
MATCH_THRESHOLD = 120.0
# Face ROI target size fed to LBPH
FACE_SIZE = (200, 200)
# Minimum detectable face area (pixels²) to ignore tiny detections
MIN_FACE_AREA = 5000


def _detect_largest_face(gray: np.ndarray):
    """Return (x, y, w, h) of the largest detected face, or None."""
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    # Pick largest by area
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = best
    if w * h < MIN_FACE_AREA:
        return None
    return (x, y, w, h)


def _extract_face_roi(frame_bgr: np.ndarray):
    """Detect face in frame, return resized grayscale ROI or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rect = _detect_largest_face(gray)
    if rect is None:
        return None
    x, y, w, h = rect
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, FACE_SIZE)
    # Equalise histogram to be robust to lighting changes
    roi = cv2.equalizeHist(roi)
    return roi


def _model_path(username: str) -> str:
    safe = username.replace("/", "_").replace("\\", "_")
    return os.path.join(MODELS_DIR, f"{safe}_face.pkl")


class FaceRecognizer:
    """Per-student face enrollment and 1:1 verification using OpenCV LBPH."""

    def enroll(self, username: str, *frames_bgr) -> bool:
        """
        Enroll a student from one or more BGR frames.
        Returns True if at least one face was found and model saved.
        """
        rois = []
        for frame in frames_bgr:
            roi = _extract_face_roi(frame)
            if roi is not None:
                rois.append(roi)
        if not rois:
            print(f"[FaceRecog] Enroll failed for '{username}' — no face detected")
            return False

        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=10, grid_x=8, grid_y=8
        )
        labels  = np.array([0] * len(rois), dtype=np.int32)
        faces   = np.array(rois, dtype=np.uint8)
        recognizer.train([faces[i] for i in range(len(faces))], labels)

        # Pickle the trained recognizer state
        tmp_xml = _model_path(username) + ".xml"
        recognizer.save(tmp_xml)
        with open(tmp_xml, "rb") as f:
            xml_bytes = f.read()
        os.remove(tmp_xml)

        data = {"xml": xml_bytes, "username": username, "samples": len(rois)}
        with open(_model_path(username), "wb") as f:
            pickle.dump(data, f)

        print(f"[FaceRecog] Enrolled '{username}' with {len(rois)} samples.")
        return True

    def is_enrolled(self, username: str) -> bool:
        return os.path.exists(_model_path(username))

    def verify(self, username: str, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Compare frame against enrolled face.
        Returns (match: bool, confidence: float).
        Lower confidence = better match; threshold = MATCH_THRESHOLD.
        Returns (False, 999) if not enrolled or no face found.
        """
        path = _model_path(username)
        if not os.path.exists(path):
            return (False, 999.0)

        with open(path, "rb") as f:
            data = pickle.load(f)

        # Reload recognizer from pickled XML
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=10, grid_x=8, grid_y=8
        )
        tmp_xml = path + "_tmp.xml"
        with open(tmp_xml, "wb") as f:
            f.write(data["xml"])
        recognizer.read(tmp_xml)
        os.remove(tmp_xml)

        roi = _extract_face_roi(frame_bgr)
        if roi is None:
            return (False, 999.0)

        label, confidence = recognizer.predict(roi)
        match = confidence <= MATCH_THRESHOLD
        print(f"[FaceRecog] Verify '{username}': label={label} confidence={confidence:.1f} → {'MATCH' if match else 'MISMATCH'}")
        return (match, float(confidence))

    def delete_enrollment(self, username: str) -> bool:
        """Remove stored face model for a student."""
        path = _model_path(username)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
