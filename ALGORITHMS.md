# ScoreHunt AI Proctorer - Technical Algorithm Reference

This document explains the core AI and behavioral analysis algorithms used to maintain exam integrity.

## 1. Computer Vision Pipeline (MediaPipe)

The system uses **MediaPipe** for high-efficiency, real-time tracking.

### A. Face Detection & Tracking
- **Algorithm**: BlazeFace (Short-range).
- **Function**: Detects the presence of 1 or more faces.
- **Cheating Detection**: Flags a violation if:
  - 0 faces are detected (Examinee left the seat).
  - 2+ faces are detected (Unauthorized person present).

### B. Gaze & Iris Tracking
- **Algorithm**: MediaPipe Iris / Face Mesh.
- **Logic**: Extracts 468 3D face landmarks to calculate the **Head Pose** and **Iris Position**.
- **Cheating Detection**: Calculate Euclidean distance and angle of the iris. If the gaze remains outside the screen boundaries for >3 seconds, a "Looking Away" warning is issued.

### C. Object Detection (Phone & Book)
- **Algorithm**: YOLOv8 (Compact Version) / MobileNet SSD.
- **Function**: Scans the frame for specific classes (`cell phone`, `book`, `laptop`).
- **Logic**: Filters out `paper` to allow scratch notes, while strictly flagging `cell phone` and `electronic devices`.

---

## 2. Keystroke Dynamics Analysis

The system implements biometric typing pattern recognition as a behavioral layer.

### A. Feature Extraction
- **Dwell Time (Hold Time)**: Duration between `KeyDown` and `KeyUp` for a specific key.
- **Flight Time (Interval)**: Duration between `KeyUp` of Key N and `KeyDown` of Key N+1.

### B. Anomaly Detection
- **Baseline**: Established during the initial "Environment Check".
- **Algorithm**: Mean Absolute Deviation (MAD).
- **Logic**: The system compares current typing speed and rhythm (Dwell + Flight) against the baseline. If the variance exceeds **2.5x the Standard Deviation**, it flags a "Suspicious Typing Pattern" (indicative of a different person or copy-pasting).

---

## 3. Audio Analysis

### A. RMS (Root Mean Square) Thresholding
- **Function**: Measures environmental volume.
- **Cheating Detection**: If the decibel level exceeds the calibrated ambient noise floor for a sustained period, it flags a "Voice/Whisper" anomaly.

### B. Frequency Analysis (FFT)
- **Function**: Uses Fast Fourier Transform to distinguish between mechanical noise (typing) and human speech frequencies (300Hz - 3kHz).

---

## 4. OS & Environment Monitoring

### A. Tab-Switching Detection
- **API**: Browser `visibilitychange` & `window.blur`.
- **Logic**: Uses a "3-Strike Policy". After 3 switches, the exam is automatically terminated.

### B. Clipboard Monitoring
- **Logic**: Blocks the `paste` event and reports attempts to paste external content into the answer fields.

---

## 5. Behavior Scoring (ScoreHunt AlertSystem)

All violations are weighted:
- **Critical (5 points)**: Multiple faces, Mobile phone detected, Tab switch.
- **Minor (1 point)**: Looking away, suspicious noise, keyboard anomaly.

**Threshold**: Total score >= 10 points or 3 Critical violations results in **Automatic Session Termination**.
