import cv2
import numpy as np
import time
import urllib.request
import os
from datetime import datetime

from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe import Image as MpImage, ImageFormat as MpImageFormat

# ── Model paths ──────────────────────────────────────────────────────────────
_DIR = os.path.dirname(__file__)
FACE_MODEL_PATH = os.path.join(_DIR, 'face_landmarker.task')
OBJ_MODEL_PATH  = os.path.join(_DIR, 'efficientdet_lite0.tflite')

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
OBJ_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"

POSE_MODEL_PATH = os.path.join(_DIR, 'pose_landmarker_lite.task')
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

EVIDENCE_DIR = os.path.join(os.path.dirname(_DIR), 'static', 'evidence')
EVIDENCE_COOLDOWN = 3.0  # seconds between saves FOR THE SAME type

# ── Face recognition (LBPH, no dlib needed) ──────────────────────────────────
try:
    from modules.face_recog import FaceRecognizer as _FR
    _face_recognizer = _FR()
    print("[VisionMonitor] Face recognition module loaded.")
except Exception as _e:
    _face_recognizer = None
    print(f"[VisionMonitor] Face recognition unavailable: {_e}")

def _ensure_model(path, url):
    if not os.path.exists(path):
        print(f"[VisionMonitor] Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print(f"[VisionMonitor] Downloaded -> {path}")

def _ensure_evidence_dir():
    os.makedirs(EVIDENCE_DIR, exist_ok=True)


class VisionMonitor:
    # ── Tunable constants ─────────────────────────────────────────────────────
    LOOK_AWAY_SEC  = 2.5   # seconds of sustained gaze-away before flagging
    NO_FACE_SEC    = 3.0   # seconds of sustained no-face before flagging
    MISMATCH_SEC   = 15.0  # seconds of sustained face-mismatch before flagging
    MISMATCH_FRAMES = 10   # consecutive mismatch frames required (belt-and-suspenders)

    def __init__(self):
        _ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)
        _ensure_model(OBJ_MODEL_PATH, OBJ_MODEL_URL)
        _ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL)
        _ensure_evidence_dir()

        # Face Landmarker
        self.face_lm = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_base_options.BaseOptions(model_asset_path=FACE_MODEL_PATH),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
                num_faces=5,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
        )

        # Object Detector (for phone / device detection)
        self.obj_det = mp_vision.ObjectDetector.create_from_options(
            mp_vision.ObjectDetectorOptions(
                base_options=mp_base_options.BaseOptions(model_asset_path=OBJ_MODEL_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                max_results=10,
                score_threshold=0.35,
            )
        )

        # Pose Landmarker (Body & Hands)
        self.pose_lm = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_base_options.BaseOptions(model_asset_path=POSE_MODEL_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
        )

        # State
        self.current_status      = "Normal"
        self.face_count          = 0
        self.is_looking_away     = False
        self.look_away_start     = None
        self.evidence_files      = []
        self._type_last_ts: dict = {}   # per-type cooldown tracking

        # ── Debounce timers ───────────────────────────────────────────────────
        self._no_face_since:     float | None = None
        self._mismatch_since:    float | None = None
        self._mismatch_frames:   int          = 0   # consecutive mismatch frame count
        # Current enrolled student (set by app.py when exam starts)
        self._enrolled_username: str | None = None


    def set_enrolled_username(self, username: str):
        """Called by app.py when a student starts their exam."""
        self._enrolled_username = username
        print(f"[VisionMonitor] Enrolled username set to '{username}'")

    # ── Evidence capture ─────────────────────────────────────────────────────
    def _save_evidence(self, frame, label):
        """Save evidence for a given label type; each type has its own 3-s cooldown."""
        now  = time.time()
        last = self._type_last_ts.get(label, 0)
        if now - last < EVIDENCE_COOLDOWN:
            return None
        self._type_last_ts[label] = now

        ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
        fname = f"evidence_{ts}_{label[:20].replace(' ', '_')}.jpg"
        path  = os.path.join(EVIDENCE_DIR, fname)
        cv2.imwrite(path, frame)
        self.evidence_files.append(fname)
        print(f"[Evidence] Saved: {fname}")
        return fname

    def capture_event_frame(self, label: str):
        """
        Grab a fresh camera frame for a browser-triggered event (e.g. tab switch).
        Returns the evidence filename or None if camera not available.
        """
        try:
            cap = cv2.VideoCapture(0)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
                safe  = label[:24].replace(' ', '_').replace('/', '_')
                fname = f"evidence_{ts}_{safe}.jpg"
                cv2.putText(frame, f"TAB SWITCH DETECTED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                cv2.putText(frame, label[:60], (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imwrite(os.path.join(EVIDENCE_DIR, fname), frame)
                print(f"[Evidence] Event frame saved: {fname}")
                return fname
        except Exception as e:
            print(f"[Evidence] capture_event_frame error: {e}")
        return None


    # ── Main processing ───────────────────────────────────────────────────────
    def process_frame(self, frame):
        infractions = []
        evidence    = None
        now         = time.time()
        mp_img = MpImage(image_format=MpImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ── 1. Face landmark detection ────────────────────────────────────────
        face_results = self.face_lm.detect(mp_img)

        if face_results.face_landmarks:
            self.face_count   = len(face_results.face_landmarks)
            self._no_face_since = None  # Reset no-face timer

            if self.face_count > 1:
                msg = f"Multiple people detected ({self.face_count})"
                infractions.append(msg)
                self.current_status = f"⚠ Multiple People ({self.face_count})"
                evidence = self._save_evidence(frame, "multiple_people")

            else:
                # ── Head-pose estimation ──────────────────────────────────────
                direction = "Forward"
                if face_results.facial_transformation_matrixes:
                    mat = np.array(
                        face_results.facial_transformation_matrixes[0].data
                    ).reshape(4, 4)
                    sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
                    if sy > 1e-6:
                        x_ang = np.degrees(np.arctan2(mat[2, 1], mat[2, 2]))
                        y_ang = np.degrees(np.arctan2(-mat[2, 0], sy))
                    else:
                        x_ang = np.degrees(np.arctan2(-mat[1, 2], mat[1, 1]))
                        y_ang = np.degrees(np.arctan2(-mat[2, 0], 0))

                    if   y_ang < -15: direction = "Looking Left"
                    elif y_ang >  15: direction = "Looking Right"
                    elif x_ang < -15: direction = "Looking Down"
                    elif x_ang >  15: direction = "Looking Up"

                if direction != "Forward":
                    if not self.is_looking_away:
                        self.is_looking_away = True
                        self.look_away_start  = now
                    elif (now - self.look_away_start) > self.LOOK_AWAY_SEC:
                        msg = f"Looking away ({direction})"
                        infractions.append(msg)
                        self.current_status = f"⚠ {direction}"
                        evidence = self._save_evidence(frame, direction)
                else:
                    self.is_looking_away = False
                    self.look_away_start  = None
                    if not self._mismatch_since:
                        self.current_status  = "Normal"

                color = (0, 255, 0) if direction == "Forward" else (0, 165, 255)
                cv2.putText(frame, f'Pose: {direction}', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # ── Face Recognition ─────────────────────────────────────────
                if _face_recognizer and self._enrolled_username:
                    if not _face_recognizer.is_enrolled(self._enrolled_username):
                        # Student hasn't enrolled yet — skip silently
                        print(f"[FaceRecog] '{self._enrolled_username}' not enrolled — skipping verify")
                        self._mismatch_since  = None
                        self._mismatch_frames = 0
                    else:
                        match, conf = _face_recognizer.verify(
                            self._enrolled_username, frame
                        )
                        if match:
                            # Confirmed same person — reset all mismatch state
                            self._mismatch_since  = None
                            self._mismatch_frames = 0
                            if self.current_status == "⚠ Face Mismatch!":
                                self.current_status = "Normal"
                            cv2.putText(frame, f"ID OK ({conf:.0f})", (20, 130),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 80), 2)
                        elif conf < 999:  # face detected but over threshold
                            self._mismatch_frames += 1
                            if self._mismatch_since is None:
                                self._mismatch_since = now
                            time_ok  = (now - self._mismatch_since) > self.MISMATCH_SEC
                            frame_ok = self._mismatch_frames >= self.MISMATCH_FRAMES
                            if time_ok and frame_ok:
                                msg = f"FACE_MISMATCH: Unrecognised person detected in exam (conf={conf:.0f})"
                                infractions.append(msg)
                                self.current_status = "⚠ Face Mismatch!"
                                evidence = self._save_evidence(frame, "face_mismatch")
                                cv2.putText(frame, f"IDENTITY ALERT! conf={conf:.0f}",
                                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.85, (0, 0, 255), 2)
                                # Reset debounce so next cycle starts fresh
                                self._mismatch_since  = None
                                self._mismatch_frames = 0
                        else:
                            # conf==999 means no face ROI found by Haar — already
                            # handled by no-face debounce above; don't double-flag
                            self._mismatch_frames = 0

            cv2.putText(frame, f'Faces: {self.face_count}', (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        else:
            # ── No face detected — debounced ──────────────────────────────────
            self.face_count       = 0
            if self._no_face_since is None:
                self._no_face_since = now
            elif (now - self._no_face_since) > self.NO_FACE_SEC:
                self.current_status = "⚠ No Face"
                infractions.append("No person detected in frame")
                evidence = self._save_evidence(frame, "no_face")
            cv2.putText(frame, 'No Face Detected', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # ── 2. Object detection (phone / book / device) ───────────────────────
        obj_results = self.obj_det.detect(mp_img)

        PHONE_LABELS = {"cell phone", "mobile phone", "phone", "remote", "laptop"}
        BOOK_LABELS  = {"book"}

        for det in obj_results.detections:
            for cat in det.categories:
                label = cat.category_name.lower()
                score = cat.score
                bb    = det.bounding_box

                if label in PHONE_LABELS and score >= 0.35:
                    msg = f"Phone/device detected ({cat.category_name}: {score:.0%})"
                    infractions.append(msg)
                    self.current_status = "⚠ Phone Detected!"
                    cv2.rectangle(frame,
                                  (bb.origin_x, bb.origin_y),
                                  (bb.origin_x + bb.width, bb.origin_y + bb.height),
                                  (0, 0, 255), 3)
                    cv2.putText(frame, f"PHONE {score:.0%}",
                                (bb.origin_x, max(bb.origin_y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    evidence = self._save_evidence(frame, "phone_detected")

                elif label in BOOK_LABELS and score >= 0.35:
                    msg = f"Book detected in frame ({score:.0%}) — possible cheating material"
                    infractions.append(msg)
                    self.current_status = "⚠ Book Detected!"
                    cv2.rectangle(frame,
                                  (bb.origin_x, bb.origin_y),
                                  (bb.origin_x + bb.width, bb.origin_y + bb.height),
                                  (0, 140, 255), 3)
                    cv2.putText(frame, f"BOOK {score:.0%}",
                                (bb.origin_x, max(bb.origin_y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
                    evidence = self._save_evidence(frame, "book_detected")

        # ── 3. Pose & Hand Detection (Body Language) ──────────────────────────────
        pose_results = self.pose_lm.detect(mp_img)
        if pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
            landmarks = pose_results.pose_landmarks[0]
            
            # 11: left shoulder, 12: right shoulder
            # 15: left wrist, 16: right wrist
            ls = landmarks[11]
            rs = landmarks[12]
            lw = landmarks[15]
            rw = landmarks[16]
            
            h, w, _ = frame.shape
            adjusted_angle = 0.0
            
            # Posture checks (Shoulder Alignment)
            if ls.visibility > 0.5 and rs.visibility > 0.5:
                dx = (ls.x - rs.x) * w
                dy = (ls.y - rs.y) * h
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Draw shoulders
                cv2.line(frame, (int(ls.x * w), int(ls.y * h)), (int(rs.x * w), int(rs.y * h)), (255, 255, 0), 2)
                
                # Normal shoulder angle is around 0 or 180 depending on orientation. Let's say +/- 20 degrees
                # If they lean heavily to one side
                adjusted_angle = abs(angle)
                if adjusted_angle > 90: adjusted_angle = 180 - adjusted_angle
                
                if adjusted_angle > 25:
                    msg = f"Posture Anomaly: Heavy leaning detected ({adjusted_angle:.1f}°)"
                    infractions.append(msg)
                    self.current_status = "⚠ Body Posture Anomaly"
                    cv2.putText(frame, 'POSTURE WARNING', (int(rs.x * w), max(10, int(rs.y * h) - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    evidence = self._save_evidence(frame, "posture_anomaly")
            
            # Hand checks (Wrists Hidden)
            # If wrists are continually out of frame (visibility < 0.2), they might be holding a phone under the desk
            hands_hidden = False
            if lw.visibility < 0.2 and rw.visibility < 0.2:
                hands_hidden = True
                cv2.putText(frame, 'HANDS HIDDEN', (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                # In a strict baseline we could flag this, for now we just log it as a minor anomaly if they lean while hidden
                if adjusted_angle > 15:  # Leaning + Hands Hidden = Suspicious
                    msg = "Suspicious Behavior: Hands hidden while leaning"
                    if msg not in infractions: infractions.append(msg)
                    self.current_status = "⚠ Suspicious Movement"
                    evidence = self._save_evidence(frame, "suspicious_movement")

        return frame, infractions, evidence   # evidence = filename or None
