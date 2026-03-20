from flask import Flask, render_template, Response, jsonify, request, send_from_directory, redirect, url_for, session
import os
import cv2
import threading
import time
import json
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

from modules.vision import VisionMonitor
from modules.audio import AudioMonitor
from modules.os_monitor import OSMonitor

app = Flask(__name__)
app.secret_key = "AI_PROCTOR_SECRET_KEY"  # for sessions

# ── Global state ──────────────────────────────────────────────────────────────
cheat_stats = {
    "warnings":     0,
    "tab_switches": 0,   # separate counter for tab-switch violations
    "events":       [],
    "evidence":     [],  # list of evidence filenames
}
is_exam_active   = False
AUTO_STOP_LIMIT  = 15   # auto-stop after this many general warnings
TAB_SWITCH_LIMIT = 3    # terminate after this many tab-switch violations

vision_monitor = VisionMonitor()
audio_monitor  = None
os_monitor     = None

EVIDENCE_DIR  = os.path.join('static', 'evidence')
SESSIONS_DIR  = os.path.join('static', 'sessions')
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
        if 'logged_in' not in session or session.get('role') != 'admin':
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def student_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or session.get('role') != 'student':
            return redirect(url_for('student_login'))
        return f(*args, **kwargs)
    return decorated_function

# ── Logging ───────────────────────────────────────────────────────────────────
def log_infraction(msg, evidence_file=None):
    global cheat_stats, is_exam_active
    print(f"[!] INCIDENT: {msg}")
    event = {"timestamp": int(time.time()), "msg": msg}
    cheat_stats["events"].append(event)
    cheat_stats["warnings"] += 1

    if evidence_file:
        cheat_stats["evidence"].append({
            "file": evidence_file,
            "msg":  msg,
            "timestamp": int(time.time()),
        })

    # Keep last 50 events
    if len(cheat_stats["events"]) > 50:
        cheat_stats["events"] = cheat_stats["events"][-50:]

    # Auto-stop if threshold hit
    if cheat_stats["warnings"] >= AUTO_STOP_LIMIT and is_exam_active:
        print(f"[!] AUTO-STOP: {AUTO_STOP_LIMIT} warnings reached.")
        stop_exam()

def save_session():
    """Persist completed exam stats to a timestamped JSON file."""
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"session_{ts}.json"
    path  = os.path.join(SESSIONS_DIR, fname)
    data  = {
        "id":         ts,
        "filename":   fname,
        "ended_at":   int(time.time()),
        "student":    cheat_stats.get("student", "Unknown"),
        "warnings":   cheat_stats["warnings"],
        "events":     cheat_stats["events"],
        "evidence":   cheat_stats["evidence"],
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
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
def generate_frames():
    global is_exam_active
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        success, frame = cap.read()
        if not success:
            break

        if is_exam_active:
            processed, infractions, evidence = vision_monitor.process_frame(frame)
            for inf in infractions:
                log_infraction(f"Vision: {inf}", evidence)
            ret, buffer = cv2.imencode('.jpg', processed)
        else:
            cv2.putText(frame, "Waiting to Start Exam", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check')
def check():
    return render_template('check.html')

@app.route('/exam')
@student_required
def exam():
    return render_template('exam.html')

@app.route('/results')
@student_required
def results():
    return render_template('results.html')

@app.route('/student')
@student_required
def student():
    return render_template('exam.html')

@app.route('/student/register', methods=['GET', 'POST'])
def student_register():
    """Register a new student account."""
    users = load_users()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users:
            return render_template('student_signup.html', error="User already exists")
        users[username] = {
            "password": generate_password_hash(password),
            "role": "student"
        }
        save_users(users)
        return redirect(url_for('student_login'))
    return render_template('student_signup.html')

@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    """Login flow for students."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users = load_users()
        user_data = users.get(username)
        if user_data and user_data.get('role') == 'student' and check_password_hash(user_data['password'], password):
            session['logged_in'] = True
            session['username'] = username
            session['role'] = 'student'
            return redirect(url_for('student'))
        return render_template('student_login.html', error="Invalid student credentials")
    return render_template('student_login.html')

@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    """Register the first/new admin account."""
    users = load_users()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users:
            return render_template('admin_signup.html', error="User already exists")
        users[username] = {
            "password": generate_password_hash(password),
            "role": "admin"
        }
        save_users(users)
        return redirect(url_for('admin_login'))
    return render_template('admin_signup.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Login flow for admin."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users = load_users()
        user_data = users.get(username)
        if user_data and user_data.get('role') == 'admin' and check_password_hash(user_data['password'], password):
            session['logged_in'] = True
            session['username'] = username
            session['role'] = 'admin'
            return redirect(url_for('admin'))
        return render_template('admin_login.html', error="Invalid credentials")
    return render_template('admin_login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/admin/logout')
def admin_logout():
    return redirect(url_for('logout'))

@app.route('/admin')
@admin_required
def admin():
    return render_template('admin.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    return jsonify({
        "warnings":          cheat_stats["warnings"],
        "tab_switches":      cheat_stats.get("tab_switches", 0),
        "tab_switch_limit":  TAB_SWITCH_LIMIT,
        "events":            cheat_stats["events"],
        "evidence":          cheat_stats["evidence"],
        "is_active":         is_exam_active,
        "student":           cheat_stats.get("student", ""),
        "vision_status":     vision_monitor.current_status if vision_monitor else "Offline",
        "audio_status":      "Active" if (audio_monitor and audio_monitor.running) else "Offline",
        "os_status":         "Active" if (os_monitor and os_monitor.running) else "Offline",
        "keystroke_status":  os_monitor.keystroke_status if os_monitor else "Offline",
        "auto_stop_limit":   AUTO_STOP_LIMIT,
    })

@app.route('/api/summary')
def get_summary():
    """Build a rich exam summary report."""
    events = cheat_stats["events"]
    evidence = cheat_stats["evidence"]
    warnings = cheat_stats["warnings"]

    # Severity classification
    HIGH_KEYWORDS = ["phone", "multiple people", "auto-stop"]
    MED_KEYWORDS  = ["looking", "audio", "window"]

    def classify(msg):
        m = msg.lower()
        if any(k in m for k in HIGH_KEYWORDS): return "high"
        if any(k in m for k in MED_KEYWORDS):  return "medium"
        return "low"

    timeline = [
        {**ev, "severity": classify(ev["msg"])}
        for ev in events
    ]

    # Verdict
    high_count = sum(1 for t in timeline if t["severity"] == "high")
    if warnings == 0:
        verdict = "pass"
    elif high_count >= 3 or warnings >= AUTO_STOP_LIMIT:
        verdict = "fail"
    else:
        verdict = "review"

    return jsonify({
        "total_warnings":    warnings,
        "total_evidence":    len(evidence),
        "high_severity":     high_count,
        "medium_severity":   sum(1 for t in timeline if t["severity"] == "medium"),
        "low_severity":      sum(1 for t in timeline if t["severity"] == "low"),
        "verdict":           verdict,
        "timeline":          timeline,
        "evidence":          evidence,
        "submitted":         cheat_stats.get("submitted", False),
    })

@app.route('/api/submit', methods=['POST'])
def submit_report():
    """Mark the exam report as submitted."""
    cheat_stats["submitted"] = True
    print("[Admin] Exam report submitted.")
    return jsonify({"status": "submitted"})


@app.route('/api/control', methods=['POST'])
def control_exam():
    global is_exam_active, audio_monitor, os_monitor, cheat_stats

    data = request.json
    if data.get('action') == 'start':
        is_exam_active = True
        cheat_stats = {"warnings": 0, "tab_switches": 0, "events": [], "evidence": [],
                       "student": data.get('student', 'Unknown')}

        if audio_monitor is None or not audio_monitor.running:
            audio_monitor = AudioMonitor(
                callback=lambda msg: log_infraction(f"Audio: {msg}"))
            audio_monitor.start()

        if os_monitor is None or not os_monitor.running:
            os_monitor = OSMonitor(
                callback=lambda msg: log_infraction(f"OS: {msg}"))
            os_monitor.start()

        return jsonify({"status": "Started"})

    elif data.get('action') == 'stop':
        stop_exam()
        return jsonify({"status": "Stopped"})

    return jsonify({"error": "Invalid action"}), 400

@app.route('/static/evidence/<filename>')
def serve_evidence(filename):
    return send_from_directory(EVIDENCE_DIR, filename)

@app.route('/api/evidence')
def list_evidence():
    return jsonify(cheat_stats["evidence"])

@app.route('/api/evidence/<filename>', methods=['DELETE'])
def delete_evidence(filename):
    """Delete an evidence file from disk and the current session list."""
    # 1. Remove from current session list
    cheat_stats["evidence"] = [e for e in cheat_stats["evidence"] if e["file"] != filename]
    
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

@app.route('/api/set_baseline', methods=['POST'])
def set_baseline():
    """Store browser baseline and apply it immediately to os_monitor."""
    data = request.json or {}
    key_count  = int(data.get('keyCount', 0))
    duration   = int(data.get('duration', 30))
    student    = data.get('student', 'Unknown')
    cheat_stats["browser_baseline"] = {
        "student":    student,
        "keyCount":   key_count,
        "mouseMoves": data.get('mouseMoves', 0),
        "duration":   duration,
    }
    # Wire baseline into running os_monitor immediately
    if os_monitor and os_monitor.running:
        os_monitor.set_browser_baseline(key_count, duration)
    print(f"[Baseline] Set for {student}: {key_count} keys/{duration}s")
    return jsonify({"status": "ok"})

@app.route('/api/report_keystroke', methods=['POST'])
def report_keystroke():
    """Receive per-keystroke dwell/flight timings from the exam page."""
    if not is_exam_active:
        return jsonify({"status": "ignored"})
    data       = request.json or {}
    dwell_ms   = float(data.get('dwell_ms', 0))
    flight_ms  = float(data.get('flight_ms', 0))
    if os_monitor and os_monitor.running:
        os_monitor.receive_keystroke_event(dwell_ms, flight_ms)
    return jsonify({"status": "ok"})

@app.route('/api/report_event', methods=['POST'])
def report_event():
    """Receive browser-side events (tab switch) from the student page."""
    global cheat_stats
    data = request.json or {}
    msg  = data.get('msg', 'Unknown browser event')
    if not is_exam_active:
        return jsonify({"status": "ignored"})

    is_tab_event = 'tab' in msg.lower() or 'focus' in msg.lower() or 'visibility' in msg.lower()
    is_shortcut  = 'shortcut' in msg.lower()

    if is_tab_event or is_shortcut:
        cheat_stats["tab_switches"] = cheat_stats.get("tab_switches", 0) + 1
        switch_count = cheat_stats["tab_switches"]
        evidence_file = vision_monitor.capture_event_frame(msg) if vision_monitor else None
        log_infraction(f"Browser: {msg}", evidence_file)

        if switch_count >= TAB_SWITCH_LIMIT:
            # Enough tab switches — terminate exam
            stop_exam()
            return jsonify({
                "status":       "stopped",
                "reason":       msg,
                "switch_count": switch_count,
                "limit":        TAB_SWITCH_LIMIT,
            })
        else:
            # Warn but keep exam going
            return jsonify({
                "status":       "warned",
                "switch_count": switch_count,
                "limit":        TAB_SWITCH_LIMIT,
                "remaining":    TAB_SWITCH_LIMIT - switch_count,
            })
    else:
        log_infraction(f"Browser: {msg}")

    return jsonify({"status": "logged"})

@app.route('/api/sessions')
@admin_required
def list_sessions():
    """List all saved past exam sessions, newest first."""
    sessions = []
    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if fname.endswith('.json'):
            path = os.path.join(SESSIONS_DIR, fname)
            try:
                with open(path) as f:
                    s = json.load(f)
                sessions.append({
                    "id":             s.get("id", fname),
                    "filename":       fname,
                    "ended_at":       s.get("ended_at", 0),
                    "student":        s.get("student", "Unknown"),
                    "warnings":       s.get("warnings", 0),
                    "evidence_count": len(s.get("evidence", [])),
                })
            except Exception:
                pass
    return jsonify(sessions)

@app.route('/api/sessions', methods=['DELETE'])
@admin_required
def delete_all_sessions():
    """Delete ALL saved session JSON files (full history wipe)."""
    deleted = 0
    errors  = []
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith('.json'):
            try:
                os.remove(os.path.join(SESSIONS_DIR, fname))
                deleted += 1
            except Exception as e:
                errors.append(str(e))
    print(f"[Admin] Deleted {deleted} session file(s).")
    return jsonify({"status": "deleted", "count": deleted, "errors": errors})

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
@admin_required
def delete_session(session_id):
    """Delete a single session JSON file by ID."""
    fname = f"session_{session_id}.json"
    path  = os.path.join(SESSIONS_DIR, fname)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    try:
        os.remove(path)
        print(f"[Admin] Session deleted: {fname}")
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>')
@admin_required
def get_session(session_id):
    """Return full data for a specific past session."""
    fname = f"session_{session_id}.json"
    path  = os.path.join(SESSIONS_DIR, fname)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    print("Starting AI Proctor Server...")
    app.run(debug=True, host='0.0.0.0', threaded=True, port=5000, use_reloader=False)
