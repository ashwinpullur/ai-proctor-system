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

# ── Multi-User Session Manager ───────────────────────────────────────────────
# username -> { 'stats': {}, 'active': bool, 'vision': VisionMonitor, ... }
SESSION_MANAGER = {}


def get_session_data(username):
    """Retrieve or initialize proctoring state for a specific user."""
    if not username:
        return None
    if username not in SESSION_MANAGER:
        # Generate a unique session ID for database tracking
        session_id = f"{username}_{int(time.time())}"
        SESSION_MANAGER[username] = {
            "session_id": session_id,
            "stats": {
                "warnings": 0,
                "face_mismatch_count": 0,
                "tab_switches": 0,
                "events": [],
                "evidence": [],
                "student": username,
                "started_at": int(time.time()),
            },
            "is_active": False,
            "vision": VisionMonitor(),
            "audio": None,
            "os": None,
            "cooldowns": {},  # per-user infraction cooldowns
        }
    return SESSION_MANAGER[username]


AUTO_STOP_LIMIT = 25  # auto-stop after this many distinct warnings
TAB_SWITCH_LIMIT = 3  # terminate after this many tab-switch violations


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


def log_infraction(username, msg, evidence_file=None):
    session_data = get_session_data(username)
    if not session_data:
        return
        
    stats = session_data["stats"]
    is_active = session_data["is_active"]
    cooldowns = session_data["cooldowns"]

    # Per-type deduplication — suppress if same category was logged recently
    key = _warn_type_key(msg)
    now = time.time()
    if now - cooldowns.get(key, 0) < WARN_TYPE_COOLDOWN:
        # Update evidence silently even if warn is suppressed
        if evidence_file:
            stats["evidence"].append({"file": evidence_file, "msg": msg, "timestamp": int(now)})
            # Persist to database immediately
            try:
                database.add_session_evidence(session_data["session_id"], evidence_file, msg, now)
            except Exception as e:
                print(f"[DB] Error saving evidence: {e}")
        return
    cooldowns[key] = now

    # ── Face-mismatch: dedicated 3-strike system ────────────────────────────────
    is_face_mismatch = "FACE_MISMATCH:" in msg
    if is_face_mismatch:
        stats["face_mismatch_count"] += 1
        strike = stats["face_mismatch_count"]
        print(f"[!] IDENTITY STRIKE {strike}/3 for {username}: {msg}")
        if strike >= 3 and is_active:
            print(f"[!] IDENTITY STRIKE 3 for {username} — terminating exam.")
            # Log strike then stop
            event = {"timestamp": int(now), "msg": msg, "face_mismatch_strike": strike}
            stats["events"].append(event)
            stats["warnings"] += 1
            if evidence_file:
                stats["evidence"].append({"file": evidence_file, "msg": msg, "timestamp": int(now)})
            stop_exam(username)
            return

    if evidence_file:
        stats["evidence"].append({"file": evidence_file, "msg": msg, "timestamp": int(now)})

    # ── Persist to database immediately ──────────────────────────────────────
    try:
        database.add_session_event(session_data["session_id"], msg, now)
        if evidence_file:
            database.add_session_evidence(session_data["session_id"], evidence_file, msg, now)
    except Exception as e:
        print(f"[DB] Real-time log error: {e}")

    # Keep last 50 events
    if len(stats["events"]) > 50:
        stats["events"] = stats["events"][-50:]

    # Auto-stop if threshold hit
    if stats["warnings"] >= AUTO_STOP_LIMIT and is_active:
        print(f"[!] AUTO-STOP ({username}): {AUTO_STOP_LIMIT} warnings reached.")
        stop_exam(username)



def save_session(username):
    """Persist completed exam stats to a timestamped JSON file AND SQLite DB."""
    session_data = get_session_data(username)
    if not session_data:
        return None
        
    stats = session_data["stats"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"session_{ts}_{username}.json"
    path = os.path.join(SESSIONS_DIR, fname)
    ended = int(time.time())
    started = stats.get("started_at", ended)
    
    data = {
        "id": ts,
        "filename": fname,
        "started_at": started,
        "ended_at": ended,
        "student": username,
        "warnings": stats["warnings"],
        "tab_switches": stats.get("tab_switches", 0),
        "events": stats["events"],
        "evidence": stats["evidence"],
    }
    # Save JSON (backwards-compatible)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    # Save to SQLite
    try:
        database.save_session(
            session_id=ts,
            student=username,
            ended_at=ended,
            started_at=started,
            warnings=data["warnings"],
            tab_switches=data["tab_switches"],
            events=data["events"],
            evidence=data["evidence"],
        )
    except Exception as e:
        print(f"[DB] Session save error for {username}: {e}")
    print(f"[Session] Saved {username} -> {fname}")
    return fname


def stop_exam(username):
    session_data = get_session_data(username)
    if not session_data:
        return
        
    if session_data["is_active"]:
        save_session(username)
        
    session_data["is_active"] = False
    
    if session_data["audio"]:
        session_data["audio"].stop()
        session_data["audio"] = None
        
    if session_data["os"]:
        session_data["os"].stop()
        session_data["os"] = None



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
    # Capture exam name from query param if provided
    exam_name = request.args.get("exam")
    if exam_name:
        session["current_exam"] = exam_name
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
    session_data = get_session_data(username)
    
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        # Process frame using user-specific monitor
        processed, infractions, evidence = session_data["vision"].process_frame(frame)
        
        # Save processed frame if there are infractions for evidence
        if infractions and evidence:
            path = os.path.join(EVIDENCE_DIR, evidence)
            cv2.imwrite(path, processed)

        for inf in infractions:
            log_infraction(username, f"Vision: {inf}", evidence)

        # Store for admin view
        _, buffer = cv2.imencode(".jpg", processed)
        PROCESSED_STREAMS[username] = buffer.tobytes()
        return jsonify({"status": "ok", "infractions": infractions})

    return jsonify({"error": "Decode failed"}), 500


@app.route("/api/stats")
def get_stats():
    username = request.args.get("student") or session.get("username")
    session_data = get_session_data(username)
    if not session_data:
        return jsonify({"error": "No active session"}), 404

    stats = session_data["stats"]
    return jsonify(
        {
            "warnings": stats["warnings"],
            "tab_switches": stats.get("tab_switches", 0),
            "tab_switch_limit": TAB_SWITCH_LIMIT,
            "face_mismatch_count": stats.get("face_mismatch_count", 0),
            "events": stats["events"],
            "evidence": stats["evidence"],
            "is_active": session_data["is_active"],
            "student": username,
            "vision_status": session_data["vision"].current_status if session_data["vision"] else "Offline",
            "audio_status": "Active" if (session_data["audio"] and session_data["audio"].running) else "Offline",
            "os_status": "Active" if (session_data["os"] and session_data["os"].running) else "Offline",
            "keystroke_status": session_data["os"].keystroke_status if session_data["os"] else "Offline",
            "auto_stop_limit": AUTO_STOP_LIMIT,
        }
    )


@app.route("/api/summary")
def get_summary():
    """Build a rich exam summary report."""
    username = request.args.get("student") or session.get("username")
    session_data = get_session_data(username)
    if not session_data:
        return jsonify({"error": "No session found"}), 404
        
    stats = session_data["stats"]
    events = stats["events"]
    evidence = stats["evidence"]
    warnings = stats["warnings"]

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
            "submitted": stats.get("submitted", False),
        }
    )


@app.route("/api/submit", methods=["POST"])
def submit_report():
    """Mark the exam report as submitted."""
    username = session.get("username")
    session_data = get_session_data(username)
    if session_data:
        session_data["stats"]["submitted"] = True
    print(f"[Admin] Exam report submitted for {username}.")
    return jsonify({"status": "submitted"})


# ── Active Students (live streams) ────────────────────────────────────────────
@app.route("/api/active_students")
@admin_required
def active_students():
    """Return list of students currently streaming (admin only)."""
    result = []
    # Loop over current processed streams or active sessions
    for username, frame in PROCESSED_STREAMS.items():
        sess = SESSION_MANAGER.get(username)
        stats = sess["stats"] if sess else {}
        result.append(
            {
                "username": username,
                "warnings": stats.get("warnings", 0),
                "tab_switches": stats.get("tab_switches", 0),
                "is_active": sess["is_active"] if sess else False,
            }
        )
    return jsonify(result)


# ── Exam & Question Management ────────────────────────────────────────────────
# --- Exam & Question Management (Redirected to Consolidated Section) ---
# Legacy routes removed. New ones are below.


@app.route("/api/control", methods=["POST"])
def control_exam():
    data = request.json
    username = data.get("student") or session.get("username")
    if not username:
        return jsonify({"error": "User not identified"}), 400
        
    session_data = get_session_data(username)
    action = data.get("action")

    if action == "start":
        session_data["is_active"] = True
        session_data["stats"] = {
            "warnings": 0,
            "face_mismatch_count": 0,
            "tab_switches": 0,
            "events": [],
            "evidence": [],
            "student": username,
            "started_at": int(time.time()),
        }
        session_data["cooldowns"] = {}
        
        # Tell vision monitor which student is enrolled
        if session_data["vision"]:
            session_data["vision"].set_enrolled_username(username)

        # Start user-specific hardware monitors (legacy/local support)
        if session_data["audio"] is None or not session_data["audio"].running:
            session_data["audio"] = AudioMonitor(
                callback=lambda msg: log_infraction(username, f"Audio: {msg}")
            )
            session_data["audio"].start()

        if session_data["os"] is None or not session_data["os"].running:
            session_data["os"] = OSMonitor(callback=lambda msg: log_infraction(username, f"OS: {msg}"))
            session_data["os"].start()

        print(f"[App] Exam STARTED for {username}")
        return jsonify({"status": "Started", "student": username})

    elif action == "stop":
        stop_exam(username)
        print(f"[App] Exam STOPPED for {username}")
        return jsonify({"status": "Stopped", "student": username})

    return jsonify({"error": "Invalid action"}), 400


@app.route("/static/evidence/<filename>")
def serve_evidence(filename):
    return send_from_directory(EVIDENCE_DIR, filename)


@app.route("/api/evidence")
def list_evidence():
    username = request.args.get("student") or session.get("username")
    session_data = get_session_data(username)
    if not session_data:
        return jsonify([])
    return jsonify(session_data["stats"]["evidence"])


@app.route("/api/evidence/<filename>", methods=["DELETE"])
def delete_evidence(filename):
    """Delete an evidence file from disk and the session list."""
    username = request.args.get("student") or session.get("username")
    session_data = get_session_data(username)
    if session_data:
        session_data["stats"]["evidence"] = [
            e for e in session_data["stats"]["evidence"] if e["file"] != filename
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
    student = data.get("student") or session.get("username") or "Unknown"
    
    session_data = get_session_data(student)
    session_data["stats"]["browser_baseline"] = {
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
    if session_data["os"] and session_data["os"].running:
        session_data["os"].set_browser_baseline(key_count, duration)

    print(f"[Baseline] Set for {student}: {key_count} keys/{duration}s")
    return jsonify({"status": "ok"})


@app.route("/api/report_keystroke", methods=["POST"])
def report_keystroke():
    """Receive per-keystroke dwell/flight timings from the exam page."""
    username = session.get("username")
    session_data = get_session_data(username)
    if not session_data or not session_data["is_active"]:
        return jsonify({"status": "ignored"})
        
    data = request.json or {}
    dwell_ms = float(data.get("dwell_ms", 0))
    flight_ms = float(data.get("flight_ms", 0))
    if session_data["os"] and session_data["os"].running:
        session_data["os"].receive_keystroke_event(dwell_ms, flight_ms)
    return jsonify({"status": "ok"})


@app.route("/api/report_event", methods=["POST"])
def report_event():
    """Receive browser-side events (tab switch, shortcuts) from the student page."""
    username = session.get("username")
    session_data = get_session_data(username)
    if not session_data or not session_data["is_active"]:
        return jsonify({"status": "ignored"})
        
    data = request.json or {}
    msg = data.get("msg", "Unknown browser event")
    stats = session_data["stats"]

    is_tab_event = (
        "tab" in msg.lower() or "focus" in msg.lower() or "visibility" in msg.lower()
    )
    is_shortcut = "shortcut" in msg.lower()

    if is_tab_event or is_shortcut:
        stats["tab_switches"] = stats.get("tab_switches", 0) + 1
        switch_count = stats["tab_switches"]

        # Capture evidence from the latest uploaded frame if possible
        evidence_file = None
        if username in PROCESSED_STREAMS:
            # We already have a processed frame in buffer, use it as evidence
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            evidence_file = f"evidence_{ts}_tab_switch.jpg"
            path = os.path.join(EVIDENCE_DIR, evidence_file)
            with open(path, "wb") as f:
                f.write(PROCESSED_STREAMS[username])
        
        log_infraction(username, f"Browser: {msg}", evidence_file)

        if switch_count >= TAB_SWITCH_LIMIT:
            # Enough tab switches — terminate exam
            stop_exam(username)
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
        log_infraction(username, f"Browser: {msg}")

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


# (Cleaned legacy Question/Settings routes)


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
# (Cleaned Student Profile routes)


# ── Exam Attendance ────────────────────────────────────────────────────────────


# (Cleaned Attendance route)


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


# Note: GET /api/admin/student-profiles is already handled by api_student_profiles above.


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


# ── Admin Extensions (Bulk Keys & Performance) ──────────────────────────────

@app.route("/api/admin/exams", methods=["GET", "POST"])
@admin_required
def api_admin_exams():
    if request.method == "POST":
        data = request.json or {}
        database.create_exam(
            exam_name=data.get("exam_name"),
            duration=data.get("duration", 60),
            total_marks=data.get("total_marks", 100)
        )
        return jsonify({"status": "created"}), 201
    
    exams = database.list_exams()
    return jsonify(exams)


@app.route("/api/admin/questions", methods=["GET", "POST"])
@admin_required
def api_admin_questions():
    exam_name = request.args.get("exam_name")
    if request.method == "POST":
        data = request.json or {}
        database.add_question(
            q_type=data.get("type"),
            question=data.get("question"),
            options=data.get("options"),
            correct_answer=data.get("correct_answer"),
            exam_name=data.get("exam_name")
        )
        return jsonify({"status": "created"}), 201
    
    if exam_name:
        questions = database.get_questions_by_exam(exam_name)
    else:
        questions = database.list_questions()
    return jsonify(questions)


@app.route("/api/admin/exams/bulk-keys", methods=["POST"])
@admin_required
def api_admin_bulk_keys():
    """Generate multiple exam keys at once in a fast transaction."""
    data = request.json or {}
    exam_name = data.get("exam_name")
    count = int(data.get("count", 10))
    
    if not exam_name:
        return jsonify({"error": "Exam name is required"}), 400
    
    new_keys = database.generate_bulk_keys(exam_name, count)
    return jsonify({
        "status": "success",
        "count": len(new_keys),
        "keys": new_keys
    })


def background_save_evidence(session_id, msg, frame_data):
    """Save evidence image and log event in a background thread."""
    try:
        # 1. Save Image
        filename = f"evidence_{session_id}_{int(time.time())}.jpg"
        save_path = os.path.join(EVIDENCE_DIR, filename)
        
        import base64
        img_bytes = base64.b64decode(frame_data.split(",")[1])
        with open(save_path, "wb") as f:
            f.write(img_bytes)
            
        # 2. Update Database
        database.add_evidence(session_id, filename, msg)
        database.add_event(session_id, f"[AUTO-LOG] {msg}")
    except Exception as e:
        print(f"[Thread Error] Failed to save evidence: {e}")


@app.route("/api/log-infraction", methods=["POST"])
def api_log_infraction():
    """Triggered by the frontend to log a violation with optional frame."""
    data = request.json or {}
    username = data.get("student") or session.get("user")
    msg = data.get("msg", "Generic Violation")
    frame = data.get("frame")
    
    sess = get_session_data(username)
    if not sess:
        return jsonify({"error": "No active session"}), 404

    # Run heavy I/O in the background to prevent "stuck" app
    if frame:
        threading.Thread(
            target=background_save_evidence, 
            args=(sess["session_id"], msg, frame)
        ).start()
    else:
        database.add_event(sess["session_id"], f"[AUTO-LOG] {msg}")

    return jsonify({"status": "logged"})


if __name__ == "__main__":
    print("Starting ScoreHunt AI Proctor Server...")
    app.run(debug=True, host="0.0.0.0", threaded=True, port=5000, use_reloader=False)
