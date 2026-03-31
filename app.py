from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    request,
    send_from_directory,
    redirect,
    url_for,
    session,
)
import os
import cv2
import numpy as np
import threading
import time
import json
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import db as database

from modules.vision import VisionMonitor
from modules.audio import AudioMonitor
from modules.os_monitor import OSMonitor

try:
    from modules.face_recog import FaceRecognizer

    face_recognizer = FaceRecognizer()
    print("[App] Face recognition ready.")
except Exception as _e:
    face_recognizer = None
    print(f"[App] Face recognition unavailable: {_e}")

app = Flask(__name__)
app.secret_key = "AI_PROCTOR_SECRET_KEY"  # for sessions

# Initialise SQLite DB
database.init_db()

# ── Global state ──────────────────────────────────────────────────────────────
cheat_stats = {
    "warnings": 0,
    "face_mismatch_count": 0,  # dedicated face-mismatch strike counter (max 3)
    "tab_switches": 0,
    "events": [],
    "evidence": [],
}
is_exam_active = False
AUTO_STOP_LIMIT = 25  # auto-stop after this many distinct warnings
TAB_SWITCH_LIMIT = 3  # terminate after this many tab-switch violations

vision_monitor = VisionMonitor()
audio_monitor = None
os_monitor = None

# LIVE STREAMS: map username -> raw_frame_bytes
ACTIVE_STREAMS = {}
PROCESSED_STREAMS = {}

EVIDENCE_DIR = os.path.join("static", "evidence")
SESSIONS_DIR = os.path.join("static", "sessions")
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# ── Auth Helpers ──────────────────────────────────────────────────────────────
USER_FILE = "users.json"


def load_users():
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "logged_in" not in session or session.get("role") != "admin":
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)

    return decorated_function


def student_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "logged_in" not in session or session.get("role") != "student":
            return redirect(url_for("student_login"))
        return f(*args, **kwargs)

    return decorated_function


# ── Logging ───────────────────────────────────────────────────────────────────
# Per-type cooldown prevents the same infraction type from flooding within N seconds
_WARN_COOLDOWNS: dict = {}
WARN_TYPE_COOLDOWN = 30.0  # seconds between same-type warnings (was 8s)


def _warn_type_key(msg: str) -> str:
    """Extract a short type key from a message for cooldown bucketing."""
    m = msg.lower()
    for kw in (
        "multiple people",
        "face mismatch",
        "no person",
        "looking",
        "phone",
        "book",
        "audio",
        "window switched",
        "copy-paste",
        "macro",
        "tab",
        "shortcut",
        "unusual typing",
    ):
        if kw in m:
            return kw
    return msg[:30]  # fallback: first 30 chars


def log_infraction(msg, evidence_file=None):
    global cheat_stats, is_exam_active

    # Per-type deduplication — suppress if same category was logged recently
    key = _warn_type_key(msg)
    now = time.time()
    if now - _WARN_COOLDOWNS.get(key, 0) < WARN_TYPE_COOLDOWN:
        # Update evidence silently even if warn is suppressed
        if evidence_file:
            cheat_stats["evidence"].append(
                {
                    "file": evidence_file,
                    "msg": msg,
                    "timestamp": int(now),
                }
            )
        return
    _WARN_COOLDOWNS[key] = now

    # ── Face-mismatch: dedicated 3-strike system ────────────────────────────────
    is_face_mismatch = "FACE_MISMATCH:" in msg
    if is_face_mismatch:
        cheat_stats["face_mismatch_count"] += 1
        strike = cheat_stats["face_mismatch_count"]
        print(f"[!] IDENTITY STRIKE {strike}/3: {msg}")
        if strike >= 3 and is_exam_active:
            print("[!] IDENTITY STRIKE 3 — terminating exam.")
            # Log all 3 strikes then stop
            event = {"timestamp": int(now), "msg": msg, "face_mismatch_strike": strike}
            cheat_stats["events"].append(event)
            cheat_stats["warnings"] += 1
            if evidence_file:
                cheat_stats["evidence"].append(
                    {
                        "file": evidence_file,
                        "msg": msg,
                        "timestamp": int(now),
                    }
                )
            stop_exam()
            return

    print(f"[!] INCIDENT: {msg}")
    event = {"timestamp": int(now), "msg": msg}
    cheat_stats["events"].append(event)
    cheat_stats["warnings"] += 1

    if evidence_file:
        cheat_stats["evidence"].append(
            {
                "file": evidence_file,
                "msg": msg,
                "timestamp": int(now),
            }
        )

    # Keep last 50 events
    if len(cheat_stats["events"]) > 50:
        cheat_stats["events"] = cheat_stats["events"][-50:]

    # Auto-stop if threshold hit
    if cheat_stats["warnings"] >= AUTO_STOP_LIMIT and is_exam_active:
        print(f"[!] AUTO-STOP: {AUTO_STOP_LIMIT} warnings reached.")
        stop_exam()


def save_session():
    """Persist completed exam stats to a timestamped JSON file AND SQLite DB."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"session_{ts}.json"
    path = os.path.join(SESSIONS_DIR, fname)
    ended = int(time.time())
    started = cheat_stats.get("started_at", ended)
    data = {
        "id": ts,
        "filename": fname,
        "started_at": started,
        "ended_at": ended,
        "student": cheat_stats.get("student", "Unknown"),
        "warnings": cheat_stats["warnings"],
        "tab_switches": cheat_stats.get("tab_switches", 0),
        "events": cheat_stats["events"],
        "evidence": cheat_stats["evidence"],
    }
    # Save JSON (backwards-compatible)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    # Save to SQLite
    try:
        database.save_session(
            session_id=ts,
            student=data["student"],
            ended_at=ended,
            started_at=started,
            warnings=data["warnings"],
            tab_switches=data["tab_switches"],
            events=data["events"],
            evidence=data["evidence"],
        )
    except Exception as e:
        print(f"[DB] Session save error: {e}")
    print(f"[Session] Saved -> {fname}")
    return fname


def stop_exam():
    global is_exam_active, audio_monitor, os_monitor
    if is_exam_active:
        save_session()
    is_exam_active = False
    if audio_monitor:
        audio_monitor.stop()
        audio_monitor = None
    if os_monitor:
        os_monitor.stop()
        os_monitor = None


# ── Video feed ────────────────────────────────────────────────────────────────
def generate_frames(username=None):
    """Serve processed frames for a specific student."""
    while True:
        if username and username in PROCESSED_STREAMS:
            frame_bytes = PROCESSED_STREAMS[username]
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        else:
            # Placeholder if no stream
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                black_frame,
                "No Active Feed",
                (180, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            ret, buffer = cv2.imencode(".jpg", black_frame)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
        time.sleep(0.3)  # Avoid tight loop


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", title="ScoreHunt | Home")


@app.route("/check")
@student_required
def check():
    return render_template("check.html")


@app.route("/exam")
@student_required
def exam():
    return render_template("exam.html")


@app.route("/results")
@student_required
def results():
    return render_template("results.html")


@app.route("/student/verify_id")
@student_required
def verify_id():
    return render_template("verify_id.html")


@app.route("/student/verify_face")
@student_required
def verify_face():
    return render_template("verify_face.html")


@app.route("/student")
@student_required
def student():
    return render_template("student.html")


@app.route("/student/register", methods=["GET", "POST"])
def student_register():
    """Register a new student account."""
    users = load_users()
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        student_id = request.form.get("student_id", "")
        year = request.form.get("year", "1st Year")
        department = request.form.get("department", "")

        if username in users:
            return render_template("student_signup.html", error="User already exists")

        users[username] = {
            "password": generate_password_hash(password),
            "role": "student",
        }
        save_users(users)
        database.create_student_profile(
            username=username, student_id=student_id, year=year, department=department
        )
        return redirect(url_for("student_login"))
    return render_template("student_signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Unified login flow for students and admins."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        role = request.form.get("role", "student")  # Default to student

        users = load_users()
        user_data = users.get(username)

        if (
            user_data
            and user_data.get("role") == role
            and check_password_hash(user_data["password"], password)
        ):
            session["logged_in"] = True
            session["username"] = username
            session["role"] = role

            if role == "admin":
                return redirect(url_for("admin"))
            else:
                # Redirect student to dashboard instead of exam verification immediately
                return redirect(url_for("student"))

        return render_template("login.html", error=f"Invalid {role} credentials")

    return render_template("login.html")

@app.route("/admin/register", methods=["GET", "POST"])
def admin_register():
    """Register the first/new admin account."""
    users = load_users()
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in users:
            return render_template("admin_signup.html", error="User already exists")
        users[username] = {
            "password": generate_password_hash(password),
            "role": "admin",
        }
        save_users(users)
        return redirect(url_for("admin_login"))
    return render_template("admin_signup.html")

@app.route("/student/login", methods=["GET", "POST"])
def student_login():
    return redirect(url_for("login"))


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/admin/logout")
def admin_logout():
    return redirect(url_for("logout"))


@app.route("/admin")
@admin_required
def admin():
    return render_template("admin.html")


@app.route("/video_feed")
def video_feed():
    # Attempt to stream for current student if logged in
    user = session.get("username")
    return Response(
        generate_frames(user), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/admin/stream/<username>")
@admin_required
def admin_video_feed(username):
    return Response(
        generate_frames(username), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/stream/upload", methods=["POST"])
@student_required
def upload_frame():
    global PROCESSED_STREAMS
    file = request.files.get("frame")
    if not file:
        return jsonify({"error": "No frame"}), 400

    username = session.get("username")
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        # Process frame
        processed, infractions, evidence = vision_monitor.process_frame(frame)
        for inf in infractions:
            log_infraction(f"Vision: {inf}", evidence)

        # Store for admin view
        _, buffer = cv2.imencode(".jpg", processed)
        PROCESSED_STREAMS[username] = buffer.tobytes()
        return jsonify({"status": "ok", "infractions": infractions})

    return jsonify({"error": "Decode failed"}), 500


@app.route("/api/stats")
def get_stats():
    return jsonify(
        {
            "warnings": cheat_stats["warnings"],
            "tab_switches": cheat_stats.get("tab_switches", 0),
            "tab_switch_limit": TAB_SWITCH_LIMIT,
            "face_mismatch_count": cheat_stats.get("face_mismatch_count", 0),
            "events": cheat_stats["events"],
            "evidence": cheat_stats["evidence"],
            "is_active": is_exam_active,
            "student": cheat_stats.get("student", ""),
            "vision_status": vision_monitor.current_status
            if vision_monitor
            else "Offline",
            "audio_status": "Active"
            if (audio_monitor and audio_monitor.running)
            else "Offline",
            "os_status": "Active" if (os_monitor and os_monitor.running) else "Offline",
            "keystroke_status": os_monitor.keystroke_status
            if os_monitor
            else "Offline",
            "auto_stop_limit": AUTO_STOP_LIMIT,
        }
    )


@app.route("/api/summary")
def get_summary():
    """Build a rich exam summary report."""
    events = cheat_stats["events"]
    evidence = cheat_stats["evidence"]
    warnings = cheat_stats["warnings"]

    # Severity classification
    HIGH_KEYWORDS = ["phone", "multiple people", "auto-stop"]
    MED_KEYWORDS = ["looking", "audio", "window"]

    def classify(msg):
        m = msg.lower()
        if any(k in m for k in HIGH_KEYWORDS):
            return "high"
        if any(k in m for k in MED_KEYWORDS):
            return "medium"
        return "low"

    timeline = [{**ev, "severity": classify(ev["msg"])} for ev in events]

    # Verdict
    high_count = sum(1 for t in timeline if t["severity"] == "high")
    if warnings == 0:
        verdict = "pass"
    elif high_count >= 3 or warnings >= AUTO_STOP_LIMIT:
        verdict = "fail"
    else:
        verdict = "review"

    return jsonify(
        {
            "total_warnings": warnings,
            "total_evidence": len(evidence),
            "high_severity": high_count,
            "medium_severity": sum(1 for t in timeline if t["severity"] == "medium"),
            "low_severity": sum(1 for t in timeline if t["severity"] == "low"),
            "verdict": verdict,
            "timeline": timeline,
            "evidence": evidence,
            "submitted": cheat_stats.get("submitted", False),
        }
    )


@app.route("/api/submit", methods=["POST"])
def submit_report():
    """Mark the exam report as submitted."""
    cheat_stats["submitted"] = True
    print("[Admin] Exam report submitted.")
    return jsonify({"status": "submitted"})


# ── Active Students (live streams) ────────────────────────────────────────────
@app.route("/api/active_students")
@admin_required
def active_students():
    """Return list of students currently streaming (admin only)."""
    result = []
    for username, frame in PROCESSED_STREAMS.items():
        result.append(
            {
                "username": username,
                "warnings": cheat_stats.get("warnings", 0)
                if cheat_stats.get("student") == username
                else 0,
                "tab_switches": cheat_stats.get("tab_switches", 0)
                if cheat_stats.get("student") == username
                else 0,
                "is_active": is_exam_active and cheat_stats.get("student") == username,
            }
        )
    return jsonify(result)


@app.route("/api/control", methods=["POST"])
def control_exam():
    global is_exam_active, audio_monitor, os_monitor, cheat_stats

    data = request.json
    if data.get("action") == "start":
        is_exam_active = True
        cheat_stats = {
            "warnings": 0,
            "face_mismatch_count": 0,
            "tab_switches": 0,
            "events": [],
            "evidence": [],
            "student": data.get("student", "Unknown"),
            "started_at": int(time.time()),
        }
        # Reset per-type cooldown table for the fresh exam session
        _WARN_COOLDOWNS.clear()

        # Tell vision monitor which student is enrolled so it can compare faces
        enrolled_user = data.get("student", "") or session.get("username", "")
        if vision_monitor:
            vision_monitor.set_enrolled_username(enrolled_user)

        if audio_monitor is None or not audio_monitor.running:
            audio_monitor = AudioMonitor(
                callback=lambda msg: log_infraction(f"Audio: {msg}")
            )
            audio_monitor.start()

        if os_monitor is None or not os_monitor.running:
            os_monitor = OSMonitor(callback=lambda msg: log_infraction(f"OS: {msg}"))
            os_monitor.start()

        return jsonify({"status": "Started"})

    elif data.get("action") == "stop":
        stop_exam()
        return jsonify({"status": "Stopped"})

    return jsonify({"error": "Invalid action"}), 400


@app.route("/static/evidence/<filename>")
def serve_evidence(filename):
    return send_from_directory(EVIDENCE_DIR, filename)


@app.route("/api/evidence")
def list_evidence():
    return jsonify(cheat_stats["evidence"])


@app.route("/api/evidence/<filename>", methods=["DELETE"])
def delete_evidence(filename):
    """Delete an evidence file from disk and the current session list."""
    # 1. Remove from current session list
    cheat_stats["evidence"] = [
        e for e in cheat_stats["evidence"] if e["file"] != filename
    ]

    # 2. Try to delete from disk
    path = os.path.join(EVIDENCE_DIR, filename)
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"[Admin] Evidence deleted: {filename}")
            return jsonify({"status": "deleted"})
        except Exception as e:
            print(f"[Admin] Error deleting file {filename}: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"status": "removed from list, file not found on disk"})


@app.route("/api/set_baseline", methods=["POST"])
def set_baseline():
    """Store browser baseline and apply it immediately to os_monitor."""
    data = request.json or {}
    key_count = int(data.get("keyCount", 0))
    duration = int(data.get("duration", 30))
    student = data.get("student", session.get("username", "Unknown"))
    cheat_stats["browser_baseline"] = {
        "student": student,
        "keyCount": key_count,
        "mouseMoves": data.get("mouseMoves", 0),
        "duration": duration,
        "avgDwell": data.get("avgDwell", 0),
        "stdDwell": data.get("stdDwell", 0),
        "avgFlight": data.get("avgFlight", 0),
        "stdFlight": data.get("stdFlight", 0),
        "avgRate": data.get("avgRate", 0),
    }
    # Wire baseline into running os_monitor immediately
    if os_monitor and os_monitor.running:
        os_monitor.set_browser_baseline(key_count, duration)

    print(
        f"[Baseline] Set for {student}: {key_count} keys/{duration}s "
        f"dwell={data.get('avgDwell', 0):.0f}ms flight={data.get('avgFlight', 0):.0f}ms"
    )
    return jsonify({"status": "ok"})


@app.route("/api/report_keystroke", methods=["POST"])
def report_keystroke():
    """Receive per-keystroke dwell/flight timings from the exam page."""
    if not is_exam_active:
        return jsonify({"status": "ignored"})
    data = request.json or {}
    dwell_ms = float(data.get("dwell_ms", 0))
    flight_ms = float(data.get("flight_ms", 0))
    if os_monitor and os_monitor.running:
        os_monitor.receive_keystroke_event(dwell_ms, flight_ms)
    return jsonify({"status": "ok"})


@app.route("/api/report_event", methods=["POST"])
def report_event():
    """Receive browser-side events (tab switch) from the student page."""
    global cheat_stats
    data = request.json or {}
    msg = data.get("msg", "Unknown browser event")
    if not is_exam_active:
        return jsonify({"status": "ignored"})

    is_tab_event = (
        "tab" in msg.lower() or "focus" in msg.lower() or "visibility" in msg.lower()
    )
    is_shortcut = "shortcut" in msg.lower()

    if is_tab_event or is_shortcut:
        cheat_stats["tab_switches"] = cheat_stats.get("tab_switches", 0) + 1
        switch_count = cheat_stats["tab_switches"]
        username = session.get("username")

        # Capture evidence from the latest uploaded frame if possible
        evidence_file = None
        if username in PROCESSED_STREAMS:
            # We already have a processed frame in buffer, use it as evidence
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            evidence_file = f"evidence_{ts}_tab_switch.jpg"
            path = os.path.join(EVIDENCE_DIR, evidence_file)
            with open(path, "wb") as f:
                f.write(PROCESSED_STREAMS[username])
        elif vision_monitor:
            # Fallback (though we want to avoid server-side camera)
            evidence_file = vision_monitor.capture_event_frame(msg)

        log_infraction(f"Browser: {msg}", evidence_file)

        if switch_count >= TAB_SWITCH_LIMIT:
            # Enough tab switches — terminate exam
            stop_exam()
            return jsonify(
                {
                    "status": "stopped",
                    "reason": msg,
                    "switch_count": switch_count,
                    "limit": TAB_SWITCH_LIMIT,
                }
            )
        else:
            # Warn but keep exam going
            return jsonify(
                {
                    "status": "warned",
                    "switch_count": switch_count,
                    "limit": TAB_SWITCH_LIMIT,
                    "remaining": TAB_SWITCH_LIMIT - switch_count,
                }
            )
    else:
        log_infraction(f"Browser: {msg}")

    return jsonify({"status": "logged"})


@app.route("/api/sessions")
@admin_required
def list_sessions():
    """List all saved past exam sessions, newest first."""
    sessions = []
    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if fname.endswith(".json"):
            path = os.path.join(SESSIONS_DIR, fname)
            try:
                with open(path) as f:
                    s = json.load(f)
                sessions.append(
                    {
                        "id": s.get("id", fname),
                        "filename": fname,
                        "ended_at": s.get("ended_at", 0),
                        "student": s.get("student", "Unknown"),
                        "warnings": s.get("warnings", 0),
                        "evidence_count": len(s.get("evidence", [])),
                    }
                )
            except Exception:
                pass
    return jsonify(sessions)


@app.route("/api/sessions", methods=["DELETE"])
@admin_required
def delete_all_sessions():
    """Delete ALL saved session JSON files (full history wipe)."""
    deleted = 0
    errors = []
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".json"):
            try:
                os.remove(os.path.join(SESSIONS_DIR, fname))
                deleted += 1
            except Exception as e:
                errors.append(str(e))
    print(f"[Admin] Deleted {deleted} session file(s).")
    return jsonify({"status": "deleted", "count": deleted, "errors": errors})


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
@admin_required
def delete_session(session_id):
    """Delete a single session JSON file by ID."""
    fname = f"session_{session_id}.json"
    path = os.path.join(SESSIONS_DIR, fname)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    try:
        os.remove(path)
        print(f"[Admin] Session deleted: {fname}")
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def get_status():
    # We need a fallback if session is dead, but let's assume it's ScoreHunt
    return jsonify({"server": "ScoreHunt AI Proctorer", "status": "online"})


@app.route("/api/sessions/<session_id>")
@admin_required
def get_session(session_id):
    """Return full data for a specific past session."""
    fname = f"session_{session_id}.json"
    path = os.path.join(SESSIONS_DIR, fname)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


# ── User Management ───────────────────────────────────────────────────────────
@app.route("/api/users/students", methods=["GET"])
@admin_required
def get_students():
    """List all registered students."""
    users = load_users()
    students = [
        {"username": uname, "role": data.get("role")}
        for uname, data in users.items()
        if data.get("role") == "student"
    ]
    return jsonify(students)


@app.route("/api/users/<username>", methods=["DELETE"])
@admin_required
def delete_user(username):
    """Delete a user account."""
    if username == session.get("username"):
        return jsonify({"error": "Cannot delete yourself"}), 400

    users = load_users()
    if username not in users:
        return jsonify({"error": "User not found"}), 404

    del users[username]
    save_users(users)
    print(f"[Admin] User deleted: {username}")
    return jsonify({"status": "deleted"})


# ── Question Bank API ─────────────────────────────────────────────────────────
@app.route("/api/questions")
def get_questions():
    """Return all questions for the exam (students use this)."""
    questions = database.list_questions()
    if not questions:
        # Fallback: return built-in sample questions so exam stays functional
        return jsonify(
            [
                {
                    "id": 1,
                    "type": "mcq",
                    "question": "What does AI stand for?",
                    "options": [
                        "Artificial Intelligence",
                        "Automated Interface",
                        "Analytical Insight",
                        "Applied Integration",
                    ],
                    "correct_answer": 0,
                    "placeholder": None,
                    "code_prompt": None,
                }
            ]
        )
    return jsonify(questions)


@app.route("/api/admin/questions", methods=["GET"])
@admin_required
def admin_list_questions():
    """Return all questions (admin view, includes correct answers)."""
    return jsonify(database.list_questions())


@app.route("/api/admin/questions", methods=["POST"])
@admin_required
def admin_add_question():
    """Add a new question to the bank."""
    data = request.json or {}
    q_type = data.get("type", "mcq")
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question text required"}), 400

    options = data.get("options")  # list of strings for MCQ
    correct_answer = data.get("correct_answer")  # int index for MCQ
    code_prompt = data.get("code_prompt")  # for code type
    placeholder = data.get("placeholder")

    new_id = database.add_question(
        q_type=q_type,
        question=question,
        options=options,
        correct_answer=correct_answer,
        code_prompt=code_prompt,
        placeholder=placeholder,
    )
    print(f"[Admin] Question added (id={new_id}): {q_type} — {question[:40]}")
    return jsonify({"status": "created", "id": new_id}), 201


@app.route("/api/admin/questions/<int:q_id>", methods=["DELETE"])
@admin_required
def admin_delete_question(q_id):
    """Delete a question by id."""
    deleted = database.delete_question(q_id)
    if deleted:
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404


# ── Exam Settings API ──────────────────────────────────────────────────────────


@app.route("/api/admin/exam-settings", methods=["GET", "POST"])
@admin_required
def api_exam_settings():
    if request.method == "POST":
        data = request.json or {}
        exam_name = data.get("exam_name", "").strip()
        if not exam_name:
            return jsonify({"error": "Exam name required"}), 400
        database.save_exam_settings(
            exam_name=exam_name,
            duration=int(data.get("duration", 60)),
            total_marks=float(data.get("total_marks", 100)),
            passing_marks=float(data.get("passing_marks", 40)),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            is_active=1 if data.get("is_active") else 0,
        )
        return jsonify({"status": "saved"}), 201
    settings = database.list_exam_settings()
    return jsonify(settings)


@app.route("/api/exam-settings/<exam_name>", methods=["GET"])
def api_get_exam_settings(exam_name):
    settings = database.get_exam_settings(exam_name)
    if not settings:
        return jsonify({"duration": 60, "total_marks": 100})
    return jsonify(settings)


# ── Face Recognition API ──────────────────────────────────────────────────────
@app.route("/api/face/enroll", methods=["POST"])
@student_required
def face_enroll():
    """Receive one or more JPEG frames from the verify_face page and enroll the student."""
    if face_recognizer is None:
        return jsonify({"error": "Face recognition not available"}), 503

    username = session.get("username")
    files = request.files.getlist("frame")  # supports multi-frame upload
    if not files:
        return jsonify({"error": "No frames provided"}), 400

    frames = []
    for f in files:
        nparr = np.frombuffer(f.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)

    if not frames:
        return jsonify({"error": "Could not decode frames"}), 400

    success = face_recognizer.enroll(username, *frames)
    if success:
        print(f"[Face] Enrolled {username} with {len(frames)} frame(s).")
        return jsonify({"status": "enrolled", "username": username})
    else:
        return jsonify(
            {
                "error": "No face detected in frames — please ensure your face is visible and well-lit"
            }
        ), 422


@app.route("/api/face/status")
@student_required
def face_status():
    """Check whether the current student has an enrolled face model."""
    if face_recognizer is None:
        return jsonify({"enrolled": False, "reason": "unavailable"})
    username = session.get("username")
    enrolled = face_recognizer.is_enrolled(username)
    return jsonify({"enrolled": enrolled, "username": username})


@app.route("/api/face/verify", methods=["POST"])
@student_required
def face_verify_api():
    """Quick one-shot verification used on the verify_id page."""
    if face_recognizer is None:
        return jsonify({"match": True, "reason": "unavailable"})  # passthrough
    username = session.get("username")
    if not face_recognizer.is_enrolled(username):
        return jsonify({"match": False, "reason": "not_enrolled"})

    f = request.files.get("frame")
    if not f:
        return jsonify({"error": "No frame"}), 400
    nparr = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Decode failed"}), 400

    match, conf = face_recognizer.verify(username, frame)
    return jsonify({"match": match, "confidence": round(conf, 1)})


# ═══════════════════════════════════════════════════════════════════════════════
# NEW FEATURES API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Student Profiles ───────────────────────────────────────────────────────────


@app.route("/api/admin/student-profiles", methods=["GET", "POST"])
@admin_required
def api_student_profiles():
    if request.method == "POST":
        data = request.json or {}
        username = data.get("username")
        if not username:
            return jsonify({"error": "Username required"}), 400
        database.create_student_profile(
            username=username,
            student_id=data.get("student_id"),
            year=data.get("year", "1st Year"),
            department=data.get("department"),
        )
        return jsonify({"status": "created"}), 201
    profiles = database.list_student_profiles()
    return jsonify(profiles)


@app.route("/api/admin/add-student", methods=["POST"])
@admin_required
def api_add_student():
    """Add a new student account manually by admin."""
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    student_id = data.get("student_id", "").strip()
    year = data.get("year", "1st Year")
    department = data.get("department", "")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    users = load_users()
    if username in users:
        return jsonify({"error": "Username already exists"}), 400

    users[username] = {"password": generate_password_hash(password), "role": "student"}
    save_users(users)

    database.create_student_profile(
        username=username, student_id=student_id, year=year, department=department
    )

    return jsonify({"status": "created", "username": username}), 201


@app.route("/api/admin/student-profiles/<username>", methods=["PUT", "DELETE"])
@admin_required
def api_admin_student_profile(username):
    if request.method == "PUT":
        data = request.json or {}
        database.update_student_profile(
            username=username,
            student_id=data.get("student_id"),
            year=data.get("year"),
            department=data.get("department"),
        )
        return jsonify({"status": "updated"})
    database.update_student_profile(username, student_id="", year="", department="")
    return jsonify({"status": "deleted"})


# ── Exam Attendance ────────────────────────────────────────────────────────────


@app.route("/api/attendance", methods=["GET"])
@admin_required
def api_attendance():
    exam_name = request.args.get("exam_name", "default")
    stats = database.get_attendance_stats(exam_name)
    attendance = database.get_attendance(exam_name)
    return jsonify({"stats": stats, "attendance": attendance})


@app.route("/api/attendance/mark", methods=["POST"])
@student_required
def api_mark_attendance():
    data = request.json or {}
    exam_name = data.get("exam_name", "default")
    username = session.get("username")
    database.mark_attendance(exam_name, username, attempted=1)
    return jsonify({"status": "marked"})


@app.route("/api/attendance/attempted", methods=["GET"])
@admin_required
def api_attempted_list():
    exam_name = request.args.get("exam_name", "default")
    attendance = database.get_attendance(exam_name)
    attempted = [a for a in attendance if a.get("attempted") == 1]
    not_attempted = [a for a in attendance if a.get("attempted") == 0]
    return jsonify({"attempted": attempted, "not_attempted": not_attempted})


# ── Hackathons ─────────────────────────────────────────────────────────────────


@app.route("/api/admin/hackathons", methods=["GET", "POST"])
@admin_required
def api_hackathons():
    if request.method == "POST":
        data = request.json or {}
        hackathon_id = database.create_hackathon(
            title=data.get("title"),
            description=data.get("description"),
            link=data.get("link"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            created_by=session.get("username"),
        )
        return jsonify({"status": "created", "id": hackathon_id}), 201
    hackathons = database.list_hackathons(active_only=False)
    return jsonify(hackathons)


@app.route("/api/admin/hackathons/<int:hackathon_id>", methods=["PUT", "DELETE"])
@admin_required
def api_hackathon(hackathon_id):
    if request.method == "DELETE":
        database.delete_hackathon(hackathon_id)
        return jsonify({"status": "deleted"})
    data = request.json or {}
    database.update_hackathon(hackathon_id, **data)
    return jsonify({"status": "updated"})


@app.route("/api/hackathons", methods=["GET"])
def api_student_hackathons():
    hackathons = database.list_hackathons(active_only=True)
    return jsonify(hackathons)


@app.route("/api/hackathons/apply", methods=["POST"])
@student_required
def api_apply_hackathon():
    data = request.json or {}
    database.apply_hackathon(
        hackathon_id=data.get("hackathon_id"),
        student=session.get("username"),
        team_name=data.get("team_name"),
        members=data.get("members"),
    )
    return jsonify({"status": "applied"})


@app.route("/api/admin/hackathon-applications", methods=["GET"])
@admin_required
def api_hackathon_applications():
    applications = database.list_hackathon_applications()
    return jsonify(applications)


# ── Student Marks ──────────────────────────────────────────────────────────────


@app.route("/api/admin/marks", methods=["GET", "POST"])
@admin_required
def api_marks():
    if request.method == "POST":
        data = request.json or {}
        database.record_marks(
            student=data.get("student"),
            exam_name=data.get("exam_name"),
            total_marks=float(data.get("total_marks", 100)),
            obtained_marks=float(data.get("obtained_marks", 0)),
            year=data.get("year"),
            recorded_by=session.get("username"),
        )
        return jsonify({"status": "recorded"})
    year = request.args.get("year")
    exam_name = request.args.get("exam_name")
    marks = database.list_marks(year=year, exam_name=exam_name)
    return jsonify(marks)


@app.route("/api/admin/top-rankers", methods=["GET"])
@admin_required
def api_top_rankers():
    exam_name = request.args.get("exam_name", "default")
    limit = int(request.args.get("limit", 3))
    rankers = database.get_top_rankers(exam_name, limit)
    return jsonify(rankers)


@app.route("/api/admin/marks/upload", methods=["POST"])
@admin_required
def api_upload_marks():
    data = request.json or {}
    marks_data = data.get("marks", [])
    for m in marks_data:
        database.record_marks(
            student=m.get("student"),
            exam_name=m.get("exam_name", data.get("exam_name", "default")),
            total_marks=float(m.get("total_marks", 100)),
            obtained_marks=float(m.get("obtained_marks", 0)),
            year=m.get("year"),
            recorded_by=session.get("username"),
        )
    return jsonify({"status": "bulk_uploaded", "count": len(marks_data)})


# ── Notifications ──────────────────────────────────────────────────────────────


@app.route("/api/admin/notifications", methods=["GET", "POST"])
@admin_required
def api_admin_notifications():
    if request.method == "POST":
        data = request.json or {}
        notif_id = database.create_notification(
            title=data.get("title"),
            message=data.get("message"),
            type=data.get("type", "info"),
            target=data.get("target", "all"),
            created_by=session.get("username"),
        )
        return jsonify({"status": "created", "id": notif_id}), 201
    notifications = database.list_notifications()
    return jsonify(notifications)


@app.route("/api/admin/notifications/<int:notif_id>", methods=["DELETE"])
@admin_required
def api_delete_notification(notif_id):
    with database.get_db() as conn:
        conn.execute("DELETE FROM notifications WHERE id=?", (notif_id,))
    return jsonify({"status": "deleted"})


@app.route("/api/student/notifications", methods=["GET"])
@student_required
def api_student_notifications():
    username = session.get("username")
    notifications = database.list_notifications(target=username)
    return jsonify(notifications)


@app.route("/api/student/notifications/<int:notif_id>/read", methods=["POST"])
@student_required
def api_mark_notification_read(notif_id):
    database.mark_notification_read(notif_id)
    return jsonify({"status": "read"})


# ── Admin Queries / Chat ───────────────────────────────────────────────────────


@app.route("/api/student/query", methods=["POST"])
@student_required
def api_student_query():
    data = request.json or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Message required"}), 400
    username = session.get("username")
    query_id = database.create_query(username, message)

    # Create notification for admin
    profile = database.get_student_details(username)
    student_name = profile.get("full_name", username) if profile else username
    database.create_notification(
        title=f"New Query from {student_name}",
        message=f"Student '{username}' asked: {message[:100]}...",
        type="info",
        target="admin",
        created_by=username,
    )

    return jsonify({"status": "sent", "id": query_id})


@app.route("/api/student/queries", methods=["GET"])
@student_required
def api_student_queries():
    queries = database.list_queries(student=session.get("username"))
    return jsonify(queries)


@app.route("/api/admin/queries", methods=["GET"])
@admin_required
def api_admin_queries():
    queries = database.list_queries()
    unread_count = database.get_unread_query_count()
    return jsonify({"queries": queries, "unread_count": unread_count})


@app.route("/api/admin/queries/<int:query_id>/respond", methods=["POST"])
@admin_required
def api_respond_query(query_id):
    data = request.json or {}
    response = data.get("response", "").strip()
    if not response:
        return jsonify({"error": "Response required"}), 400
    database.respond_query(query_id, response)
    return jsonify({"status": "responded"})


# ── Exam Keys ─────────────────────────────────────────────────────────────────


@app.route("/api/admin/exam-key", methods=["POST"])
@admin_required
def api_set_exam_key():
    data = request.json or {}
    database.set_exam_key(
        exam_name=data.get("exam_name", "default"), answer_key=data.get("answer_key")
    )
    return jsonify({"status": "key_set"})


@app.route("/api/admin/exam-key/release", methods=["POST"])
@admin_required
def api_release_exam_key():
    data = request.json or {}
    exam_name = data.get("exam_name", "default")
    database.release_exam_key(exam_name, session.get("username"))
    return jsonify({"status": "key_released"})


@app.route("/api/student/exam-key/<exam_name>", methods=["GET"])
@student_required
def api_get_exam_key(exam_name):
    key_data = database.get_exam_key(exam_name)
    if key_data and key_data.get("released") == 1:
        return jsonify({"key": key_data.get("answer_key"), "released": True})
    return jsonify({"key": None, "released": False})


# ── Results ───────────────────────────────────────────────────────────────────


@app.route("/api/admin/results", methods=["GET", "POST"])
@admin_required
def api_admin_results():
    if request.method == "POST":
        data = request.json or {}
        if data.get("action") == "publish":
            database.publish_results(
                data.get("exam_name", "default"), session.get("username")
            )
            database.calculate_rankings(data.get("exam_name", "default"))
            return jsonify({"status": "published"})
    results = database.get_published_results()
    return jsonify(results)


@app.route("/api/student/results", methods=["GET"])
@student_required
def api_student_results():
    username = session.get("username")
    results = database.get_result(username)
    return jsonify(results)


@app.route("/api/student/year-mates-results", methods=["GET"])
@student_required
def api_year_mates_results():
    username = session.get("username")
    results = database.get_year_mates_results(username)
    return jsonify(results)


@app.route("/api/admin/students", methods=["GET"])
@admin_required
def api_admin_students():
    students_by_year = database.list_students_by_year()
    return jsonify(students_by_year)


@app.route("/api/student/profile", methods=["GET", "POST", "PUT"])
@student_required
def api_student_profile():
    username = session.get("username")
    if request.method in ["POST", "PUT"]:
        data = request.json or {}
        # Enforce mandatory fields
        mandatory = ["full_name", "year", "parent_name", "parent_phone"]
        missing = [f for f in mandatory if not data.get(f)]
        if missing:
            return jsonify({"error": f"Missing mandatory fields: {', '.join(missing)}"}), 400

        database.update_student_details(username, **data)
        return jsonify({"status": "updated"})
    
    profile = database.get_student_details(username)
    if not profile:
        database.create_student_profile(username)
        profile = database.get_student_details(username)
    return jsonify(profile)


@app.route("/api/student/profile/<username>", methods=["GET"])
@admin_required
def api_get_student_profile(username):
    profile = database.get_student_details(username)
    if not profile:
        return jsonify({"error": "Profile not found"}), 404
    return jsonify(profile)


@app.route("/api/admin/student-profiles", methods=["GET"])
@admin_required
def api_admin_student_profiles():
    # Flattening the list_students_by_year for the simple table view if needed
    by_year = database.list_students_by_year()
    all_students = []
    for year in by_year:
        all_students.extend(by_year[year])
    return jsonify(all_students)


@app.route("/api/admin/student-profiles/<username>", methods=["PUT"])
@admin_required
def api_admin_update_student_profile(username):
    data = request.json or {}
    database.update_student_details(username, **data)
    return jsonify({"status": "updated"})



@app.route("/api/active_students", methods=["GET"])
@admin_required
def api_active_students():
    # Simple implementation: list students with active sessions
    # We can use the already tracked stats or query the database
    # For now, let's return students who have a session started within the last hour
    active = database.get_active_exam_students()
    return jsonify(active)


@app.route("/api/student/result/<exam_name>", methods=["GET"])
@student_required
def api_student_result(exam_name):
    result = database.get_result(session.get("username"), exam_name)
    if not result:
        return jsonify({"error": "Result not found"}), 404
    return jsonify(result)


# ── Re-attempts ────────────────────────────────────────────────────────────────


@app.route("/api/admin/reattempts", methods=["GET", "POST"])
@admin_required
def api_admin_reattempts():
    if request.method == "POST":
        data = request.json or {}
        action = data.get("action")
        if action == "allow":
            database.allow_reattempt(
                student=data.get("student"),
                exam_name=data.get("exam_name", "default"),
                allowed_by=session.get("username"),
            )
            return jsonify({"status": "allowed"})
        elif action == "request":
            with database.get_db() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO re_attempts (student, exam_name, allowed)
                       VALUES (?,?,0)""",
                    (data.get("student"), data.get("exam_name", "default")),
                )
            return jsonify({"status": "request_created"})
    requests = database.list_reattempt_requests()
    return jsonify(requests)


@app.route("/api/student/reattempt/request", methods=["POST"])
@student_required
def api_student_reattempt_request():
    data = request.json or {}
    with database.get_db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO re_attempts (student, exam_name, allowed)
               VALUES (?,?,0)""",
            (session.get("username"), data.get("exam_name", "default")),
        )
    return jsonify({"status": "request_submitted"})


@app.route("/api/student/reattempt/check", methods=["GET"])
@student_required
def api_check_reattempt():
    exam_name = request.args.get("exam_name", "default")
    result = database.check_reattempt(session.get("username"), exam_name)
    return jsonify({"allowed": result is not None})



@app.route("/api/student/resume/upload", methods=["POST"])
@student_required
def api_upload_resume():
    username = session.get("username")
    file = request.files.get("resume")

    if not file:
        return jsonify({"error": "No file provided"}), 400

    UPLOAD_DIR = os.path.join("static", "uploads", "resumes")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    ext = os.path.splitext(file.filename)[1] if file.filename else ".pdf"
    filename = f"resume_{username}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    resume_path = f"/static/uploads/resumes/{filename}"
    database.update_student_details(username, resume_path=resume_path)

    return jsonify({"status": "uploaded", "path": resume_path})


# ── Dashboard Statistics ──────────────────────────────────────────────────────


@app.route("/api/admin/dashboard-stats", methods=["GET"])
@admin_required
def api_dashboard_stats():
    users = load_users()
    students = [u for u, d in users.items() if d.get("role") == "student"]

    with database.get_db() as conn:
        total_sessions = conn.execute(
            "SELECT COUNT(*) as cnt FROM exam_sessions"
        ).fetchone()["cnt"]
        total_marks = conn.execute(
            "SELECT COUNT(*) as cnt FROM student_marks"
        ).fetchone()["cnt"]
        active_hackathons = conn.execute(
            "SELECT COUNT(*) as cnt FROM hackathons WHERE active=1"
        ).fetchone()["cnt"]
        unread_queries = database.get_unread_query_count()

    return jsonify(
        {
            "total_students": len(students),
            "total_sessions": total_sessions,
            "total_marks_records": total_marks,
            "active_hackathons": active_hackathons,
            "unread_queries": unread_queries,
        }
    )


# ── Year-wise Statistics ──────────────────────────────────────────────────────


@app.route("/api/admin/year-stats", methods=["GET"])
@admin_required
def api_year_stats():
    with database.get_db() as conn:
        rows = conn.execute("""
            SELECT sp.year, COUNT(DISTINCT sp.username) as student_count,
                   AVG(sm.obtained_marks) as avg_marks,
                   MAX(sm.obtained_marks) as max_marks
            FROM student_profiles sp
            LEFT JOIN student_marks sm ON sp.username = sm.student
            GROUP BY sp.year
        """).fetchall()
    return jsonify([dict(r) for r in rows])


# ── Delete Evidence from Database ─────────────────────────────────────────────


@app.route("/api/admin/evidence/cleanup", methods=["POST"])
@admin_required
def api_cleanup_evidence():
    data = request.json or {}
    older_than_days = data.get("older_than_days", 30)
    cutoff_time = int(time.time()) - (older_than_days * 86400)

    with database.get_db() as conn:
        evidence_files = conn.execute(
            "SELECT filename FROM session_evidence WHERE timestamp < ?", (cutoff_time,)
        ).fetchall()

        deleted_count = 0
        for ev in evidence_files:
            filepath = os.path.join(EVIDENCE_DIR, ev["filename"])
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    print(f"[Admin] Error deleting {ev['filename']}: {e}")

        conn.execute("DELETE FROM session_evidence WHERE timestamp < ?", (cutoff_time,))

    return jsonify({"status": "cleaned", "deleted_files": deleted_count})


if __name__ == "__main__":
    print("Starting ScoreHunt AI Proctor Server...")
    app.run(debug=True, host="0.0.0.0", threaded=True, port=5000, use_reloader=False)
