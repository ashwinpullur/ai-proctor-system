"""
db.py — SQLite database module for ScoreHunt AI Proctor
Handles: questions, exam sessions, events, and evidence storage.
"""

import sqlite3
import os
import json
import time
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "proctor.db")


@contextmanager
def get_db():
    """Context manager: yields a connected SQLite connection and commits on success."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # concurrent reads
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist yet."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS questions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                type            TEXT    NOT NULL CHECK(type IN ('mcq','code','theory')),
                question        TEXT    NOT NULL,
                options         TEXT,           -- JSON array of strings (MCQ only)
                correct_answer  INTEGER,        -- 0-based index (MCQ only)
                code_prompt     TEXT,           -- starter / example code (code type)
                placeholder     TEXT,           -- textarea placeholder text
                exam_name       TEXT,           -- associated exam name
                duration        INTEGER DEFAULT 60,  -- exam duration in minutes
                created_at      INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS exam_settings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_name       TEXT UNIQUE NOT NULL,
                duration        INTEGER DEFAULT 60,
                total_marks     REAL DEFAULT 100,
                passing_marks   REAL DEFAULT 40,
                start_time      TEXT,
                end_time        TEXT,
                is_active       INTEGER DEFAULT 0,
                created_at      INTEGER DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS exam_sessions (
                id          TEXT    PRIMARY KEY,
                student     TEXT,
                started_at  INTEGER,
                ended_at    INTEGER,
                warnings    INTEGER DEFAULT 0,
                tab_switches INTEGER DEFAULT 0,
                verdict     TEXT    DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS session_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                timestamp   INTEGER NOT NULL,
                msg         TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES exam_sessions(id)
            );

            CREATE TABLE IF NOT EXISTS session_evidence (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                filename    TEXT    NOT NULL,
                msg         TEXT,
                timestamp   INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES exam_sessions(id)
            );

            CREATE TABLE IF NOT EXISTS student_profiles (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT UNIQUE NOT NULL,
                student_id      TEXT,
                full_name       TEXT,
                phone           TEXT,
                year            TEXT DEFAULT '1st Year',
                department      TEXT,
                parent_name     TEXT,
                parent_phone    TEXT,
                git_profile     TEXT,
                resume_path     TEXT,
                created_at      INTEGER DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS exam_attendance (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_name       TEXT NOT NULL,
                student         TEXT NOT NULL,
                attempted       INTEGER DEFAULT 0,
                attempted_at    INTEGER,
                status          TEXT DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS hackathons (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                title           TEXT NOT NULL,
                description     TEXT,
                link            TEXT,
                start_date      TEXT,
                end_date        TEXT,
                created_by      TEXT,
                created_at      INTEGER DEFAULT (strftime('%s','now')),
                active          INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS hackathon_applications (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                hackathon_id    INTEGER,
                student         TEXT NOT NULL,
                team_name       TEXT,
                members         TEXT,
                applied_at      INTEGER DEFAULT (strftime('%s','now')),
                status          TEXT DEFAULT 'pending',
                FOREIGN KEY (hackathon_id) REFERENCES hackathons(id)
            );

            CREATE TABLE IF NOT EXISTS student_marks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                student         TEXT NOT NULL,
                exam_name       TEXT NOT NULL,
                total_marks     REAL DEFAULT 0,
                obtained_marks  REAL DEFAULT 0,
                year            TEXT,
                recorded_by     TEXT,
                recorded_at     INTEGER DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS notifications (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                title           TEXT NOT NULL,
                message         TEXT,
                type            TEXT DEFAULT 'info',
                target          TEXT DEFAULT 'all',
                created_by      TEXT,
                created_at      INTEGER DEFAULT (strftime('%s','now')),
                is_read         INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS admin_queries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                student         TEXT NOT NULL,
                message         TEXT NOT NULL,
                sender          TEXT DEFAULT 'student',
                response        TEXT,
                responded_at    INTEGER,
                created_at      INTEGER DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS exam_keys (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_name       TEXT NOT NULL,
                key_value       TEXT UNIQUE NOT NULL,
                is_used         INTEGER DEFAULT 0,
                created_at      INTEGER DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                student         TEXT NOT NULL,
                exam_name       TEXT NOT NULL,
                marks           REAL,
                rank            INTEGER,
                total_students  INTEGER,
                percentage      REAL,
                verdict         TEXT,
                published       INTEGER DEFAULT 0,
                published_at    INTEGER,
                published_by    TEXT,
                created_at      INTEGER DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS re_attempts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                student         TEXT NOT NULL,
                exam_name       TEXT NOT NULL,
                allowed         INTEGER DEFAULT 0,
                allowed_by      TEXT,
                allowed_at      INTEGER,
                used            INTEGER DEFAULT 0,
                used_at         INTEGER
            );
        """)
    print("[DB] Database initialized ->", DB_PATH)
    _migrate_db()


def _migrate_db():
    """Safely add columns/indexes that were introduced after initial deployment.
    Uses try/except so already-applied migrations are silently skipped."""
    migrations = [
        # hackathons: active column was added later
        "ALTER TABLE hackathons ADD COLUMN active INTEGER DEFAULT 1",
        # student_profiles: extended fields added later
        "ALTER TABLE student_profiles ADD COLUMN full_name TEXT",
        "ALTER TABLE student_profiles ADD COLUMN phone TEXT",
        "ALTER TABLE student_profiles ADD COLUMN parent_name TEXT",
        "ALTER TABLE student_profiles ADD COLUMN parent_phone TEXT",
        "ALTER TABLE student_profiles ADD COLUMN git_profile TEXT",
        "ALTER TABLE student_profiles ADD COLUMN resume_path TEXT",
    ]
    with get_db() as conn:
        for sql in migrations:
            try:
                conn.execute(sql)
            except Exception:
                pass  # Column already exists — safe to ignore
    print("[DB] Migration check complete.")


# ── Questions CRUD ─────────────────────────────────────────────────────────────


def add_question(
    q_type,
    question,
    options=None,
    correct_answer=None,
    code_prompt=None,
    placeholder=None,
    exam_name=None,
):
    """Insert a new question. Returns the new row id."""
    opts_json = json.dumps(options) if options else None
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO questions
               (type, question, options, correct_answer, code_prompt, placeholder, exam_name)
               VALUES (?,?,?,?,?,?,?)""",
            (q_type, question, opts_json, correct_answer, code_prompt, placeholder, exam_name),
        )
        return cur.lastrowid


def get_questions_by_exam(exam_name):
    """Return questions belonging to a specific exam."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM questions WHERE exam_name=? ORDER BY id", (exam_name,)).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d["options"]:
            d["options"] = json.loads(d["options"])
        result.append(d)
    return result


def list_questions():
    """Return all questions as a list of dicts (options decoded from JSON)."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM questions ORDER BY id").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d["options"]:
            d["options"] = json.loads(d["options"])
        result.append(d)
    return result


def delete_question(q_id):
    """Delete a question by id. Returns True if deleted."""
    with get_db() as conn:
        cur = conn.execute("DELETE FROM questions WHERE id=?", (q_id,))
        return cur.rowcount > 0


def get_exam_settings(exam_name):
    """Get exam settings by exam name."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM exam_settings WHERE exam_name=?", (exam_name,)
        ).fetchone()
        return dict(row) if row else None


def save_exam_settings(
    exam_name,
    duration=60,
    total_marks=100,
    passing_marks=40,
    start_time=None,
    end_time=None,
    is_active=0,
):
    """Save or update exam settings."""
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO exam_settings 
               (exam_name, duration, total_marks, passing_marks, start_time, end_time, is_active)
               VALUES (?,?,?,?,?,?,?)""",
            (
                exam_name,
                duration,
                int(total_marks),
                int(passing_marks),
                start_time,
                end_time,
                is_active,
            ),
        )


def list_exams():
    """Return all defined exams."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM exam_settings ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


# ── Evidence & Events ─────────────────────────────────────────────────────────


def add_session_evidence(session_id, filename, msg, timestamp):
    """Persist an evidence record to the database."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO session_evidence (session_id, filename, msg, timestamp)
               VALUES (?,?,?,?)""",
            (session_id, filename, msg, int(timestamp)),
        )


def get_session_evidence(session_id):
    """Retrieve all evidence for a specific session."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM session_evidence WHERE session_id=? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def add_session_event(session_id, msg, timestamp):
    """Persist a live event to the database."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO session_events (session_id, msg, timestamp) VALUES (?,?,?)",
            (session_id, msg, int(timestamp)),
        )


def get_session_events(session_id):
    """Retrieve event history for a session."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM session_events WHERE session_id=? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_exam_settings():
    """List all exam settings."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM exam_settings ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def update_question_exam_info(q_id, exam_name, duration):
    """Update question with exam name and duration."""
    with get_db() as conn:
        conn.execute(
            "UPDATE questions SET exam_name=?, duration=? WHERE id=?",
            (exam_name, duration, q_id),
        )


# ── Session CRUD ───────────────────────────────────────────────────────────────


def save_session(
    session_id, student, ended_at, started_at, warnings, tab_switches, events, evidence
):
    """Upsert a completed exam session and its events/evidence."""
    # Determine verdict
    if warnings >= 15 or tab_switches >= 3:
        verdict = "fail"
    elif warnings == 0:
        verdict = "pass"
    else:
        verdict = "review"

    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO exam_sessions
               (id, student, started_at, ended_at, warnings, tab_switches, verdict)
               VALUES (?,?,?,?,?,?,?)""",
            (
                session_id,
                student,
                started_at,
                ended_at,
                warnings,
                tab_switches,
                verdict,
            ),
        )
        # Delete old events/evidence for this session before reinserting
        conn.execute("DELETE FROM session_events WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM session_evidence WHERE session_id=?", (session_id,))

        for ev in events:
            conn.execute(
                "INSERT INTO session_events (session_id, timestamp, msg) VALUES (?,?,?)",
                (session_id, ev.get("timestamp", int(time.time())), ev.get("msg", "")),
            )
        for ev in evidence:
            conn.execute(
                """INSERT INTO session_evidence (session_id, filename, msg, timestamp)
                   VALUES (?,?,?,?)""",
                (
                    session_id,
                    ev.get("file", ""),
                    ev.get("msg", ""),
                    ev.get("timestamp", int(time.time())),
                ),
            )


def list_sessions():
    """Return all sessions ordered newest-first."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT es.*, COUNT(se.id) as evidence_count
               FROM exam_sessions es
               LEFT JOIN session_evidence se ON se.session_id = es.id
               GROUP BY es.id
               ORDER BY es.ended_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id):
    """Return a single session with its events and evidence."""
    with get_db() as conn:
        s = conn.execute(
            "SELECT * FROM exam_sessions WHERE id=?", (session_id,)
        ).fetchone()
        if not s:
            return None
        events = conn.execute(
            "SELECT timestamp, msg FROM session_events WHERE session_id=? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        evidence = conn.execute(
            "SELECT filename as file, msg, timestamp FROM session_evidence WHERE session_id=?",
            (session_id,),
        ).fetchall()

    result = dict(s)
    result["events"] = [dict(e) for e in events]
    result["evidence"] = [dict(e) for e in evidence]
    return result


# ── Student Profiles ───────────────────────────────────────────────────────────


def create_student_profile(username, student_id=None, year="1st Year", department=None):
    with get_db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO student_profiles (username, student_id, year, department)
               VALUES (?,?,?,?)""",
            (username, student_id, year, department),
        )


def get_student_profile(username):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM student_profiles WHERE username=?", (username,)
        ).fetchone()
        return dict(row) if row else None


def update_student_profile(username, student_id=None, year=None, department=None):
    with get_db() as conn:
        conn.execute(
            """UPDATE student_profiles SET student_id=COALESCE(?,student_id),
               year=COALESCE(?,year), department=COALESCE(?,department) WHERE username=?""",
            (student_id, year, department, username),
        )


def update_student_details(
    username,
    full_name=None,
    phone=None,
    parent_name=None,
    parent_phone=None,
    git_profile=None,
    resume_path=None,
):
    with get_db() as conn:
        conn.execute(
            """UPDATE student_profiles SET 
               full_name=COALESCE(?,full_name),
               phone=COALESCE(?,phone),
               parent_name=COALESCE(?,parent_name),
               parent_phone=COALESCE(?,parent_phone),
               git_profile=COALESCE(?,git_profile),
               resume_path=COALESCE(?,resume_path)
               WHERE username=?""",
            (
                full_name,
                phone,
                parent_name,
                parent_phone,
                git_profile,
                resume_path,
                username,
            ),
        )


def get_student_details(username):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM student_profiles WHERE username=?", (username,)
        ).fetchone()
        return dict(row) if row else None


def list_student_profiles():
    with get_db() as conn:
        return [
            dict(r) for r in conn.execute("SELECT * FROM student_profiles").fetchall()
        ]


# ── Exam Attendance ────────────────────────────────────────────────────────────


def mark_attendance(exam_name, student, attempted=1):
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO exam_attendance (exam_name, student, attempted, attempted_at, status)
               VALUES (?,?,?,?,?)""",
            (
                exam_name,
                student,
                attempted,
                int(time.time()),
                "completed" if attempted else "missed",
            ),
        )


def get_attendance(exam_name=None):
    with get_db() as conn:
        if exam_name:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM exam_attendance WHERE exam_name=?", (exam_name,)
                ).fetchall()
            ]
        return [
            dict(r) for r in conn.execute("SELECT * FROM exam_attendance").fetchall()
        ]


def get_attendance_stats(exam_name):
    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM exam_attendance WHERE exam_name=?",
            (exam_name,),
        ).fetchone()["cnt"]
        attempted = conn.execute(
            "SELECT COUNT(*) as cnt FROM exam_attendance WHERE exam_name=? AND attempted=1",
            (exam_name,),
        ).fetchone()["cnt"]
        return {
            "total": total,
            "attempted": attempted,
            "not_attempted": total - attempted,
        }


# ── Hackathons ─────────────────────────────────────────────────────────────────


def create_hackathon(
    title,
    description=None,
    link=None,
    start_date=None,
    end_date=None,
    created_by="admin",
):
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO hackathons (title, description, link, start_date, end_date, created_by)
               VALUES (?,?,?,?,?,?)""",
            (title, description, link, start_date, end_date, created_by),
        )
        return cur.lastrowid


def list_hackathons(active_only=True):
    with get_db() as conn:
        if active_only:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM hackathons WHERE active=1 ORDER BY created_at DESC"
                ).fetchall()
            ]
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM hackathons ORDER BY created_at DESC"
            ).fetchall()
        ]


def update_hackathon(hackathon_id, **kwargs):
    with get_db() as conn:
        for key, val in kwargs.items():
            conn.execute(
                f"UPDATE hackathons SET {key}=? WHERE id=?", (val, hackathon_id)
            )


def delete_hackathon(hackathon_id):
    with get_db() as conn:
        conn.execute("UPDATE hackathons SET active=0 WHERE id=?", (hackathon_id,))


def apply_hackathon(hackathon_id, student, team_name=None, members=None):
    members_json = json.dumps(members) if members else None
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO hackathon_applications (hackathon_id, student, team_name, members)
               VALUES (?,?,?,?)""",
            (hackathon_id, student, team_name, members_json),
        )


def list_hackathon_applications(hackathon_id=None):
    with get_db() as conn:
        if hackathon_id:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM hackathon_applications WHERE hackathon_id=?",
                    (hackathon_id,),
                ).fetchall()
            ]
        return [
            dict(r)
            for r in conn.execute("SELECT * FROM hackathon_applications").fetchall()
        ]


# ── Student Marks ──────────────────────────────────────────────────────────────


def record_marks(
    student, exam_name, total_marks, obtained_marks, year=None, recorded_by="admin"
):
    percentage = (obtained_marks / total_marks * 100) if total_marks > 0 else 0
    verdict = "pass" if percentage >= 40 else "fail"
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO student_marks (student, exam_name, total_marks, obtained_marks, year, recorded_by)
               VALUES (?,?,?,?,?,?)""",
            (student, exam_name, total_marks, obtained_marks, year, recorded_by),
        )
        conn.execute(
            """INSERT OR REPLACE INTO results (student, exam_name, marks, percentage, verdict)
               VALUES (?,?,?,?,?)""",
            (student, exam_name, obtained_marks, percentage, verdict),
        )


def list_marks(student=None, exam_name=None, year=None):
    with get_db() as conn:
        query = "SELECT * FROM student_marks WHERE 1=1"
        params = []
        if student:
            query += " AND student=?"
            params.append(student)
        if exam_name:
            query += " AND exam_name=?"
            params.append(exam_name)
        if year:
            query += " AND year=?"
            params.append(year)
        query += " ORDER BY obtained_marks DESC"
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def get_top_rankers(exam_name, limit=3):
    with get_db() as conn:
        return [
            dict(r)
            for r in conn.execute(
                """SELECT sm.*, sp.student_id, sp.year, sp.department
               FROM student_marks sm
               LEFT JOIN student_profiles sp ON sm.student = sp.username
               WHERE sm.exam_name=? ORDER BY sm.obtained_marks DESC LIMIT ?""",
                (exam_name, limit),
            ).fetchall()
        ]


# ── Notifications ─────────────────────────────────────────────────────────────


def create_notification(title, message, type="info", target="all", created_by="admin"):
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO notifications (title, message, type, target, created_by)
               VALUES (?,?,?,?,?)""",
            (title, message, type, target, created_by),
        )
        return cur.lastrowid


def list_notifications(target="all"):
    with get_db() as conn:
        if target == "all":
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM notifications ORDER BY created_at DESC"
                ).fetchall()
            ]
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM notifications WHERE target IN (?, 'all') ORDER BY created_at DESC",
                (target,),
            ).fetchall()
        ]


def mark_notification_read(notification_id):
    with get_db() as conn:
        conn.execute(
            "UPDATE notifications SET is_read=1 WHERE id=?", (notification_id,)
        )


# ── Admin Queries / Chat ──────────────────────────────────────────────────────


def create_query(student, message):
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO admin_queries (student, message, sender)
               VALUES (?,?,?)""",
            (student, message, "student"),
        )
        return cur.lastrowid


def respond_query(query_id, response):
    with get_db() as conn:
        conn.execute(
            """UPDATE admin_queries SET response=?, responded_at=? WHERE id=?""",
            (response, int(time.time()), query_id),
        )


def list_queries(student=None):
    with get_db() as conn:
        if student:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM admin_queries WHERE student=? ORDER BY created_at ASC",
                    (student,),
                ).fetchall()
            ]
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM admin_queries ORDER BY created_at DESC"
            ).fetchall()
        ]


def get_unread_query_count():
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) as cnt FROM admin_queries WHERE response IS NULL AND sender='student'"
        ).fetchone()["cnt"]


# ── Exam Keys ─────────────────────────────────────────────────────────────────


def set_exam_key(exam_name, answer_key):
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO exam_keys (exam_name, answer_key, released)
               VALUES (?,?,0)""",
            (exam_name, answer_key),
        )


def release_exam_key(exam_name, released_by="admin"):
    with get_db() as conn:
        conn.execute(
            """UPDATE exam_keys SET released=1, released_at=?, released_by=? WHERE exam_name=?""",
            (int(time.time()), released_by, exam_name),
        )


def get_exam_key(exam_name):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM exam_keys WHERE exam_name=?", (exam_name,)
        ).fetchone()
        return dict(row) if row else None


# ── Results ───────────────────────────────────────────────────────────────────


def get_result(student, exam_name=None):
    with get_db() as conn:
        if exam_name:
            row = conn.execute(
                "SELECT * FROM results WHERE student=? AND exam_name=?",
                (student, exam_name),
            ).fetchone()
            return dict(row) if row else None
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM results WHERE student=? ORDER BY created_at DESC",
                (student,),
            ).fetchall()
        ]


def publish_results(exam_name, published_by="admin"):
    with get_db() as conn:
        conn.execute(
            """UPDATE results SET published=1, published_at=?, published_by=? WHERE exam_name=?""",
            (int(time.time()), published_by, exam_name),
        )


def get_published_results(exam_name=None):
    with get_db() as conn:
        if exam_name:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM results WHERE published=1 AND exam_name=? ORDER BY rank ASC",
                    (exam_name,),
                ).fetchall()
            ]
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM results WHERE published=1 ORDER BY exam_name, rank ASC"
            ).fetchall()
        ]


def calculate_rankings(exam_name):
    with get_db() as conn:
        rows = conn.execute(
            """SELECT student, marks, RANK() OVER (ORDER BY marks DESC) as rank,
               COUNT(*) OVER () as total_students
               FROM results WHERE exam_name=? AND published=0""",
            (exam_name,),
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE results SET rank=?, total_students=? WHERE student=? AND exam_name=?",
                (row["rank"], row["total_students"], row["student"], exam_name),
            )


# ── Re-attempts ────────────────────────────────────────────────────────────────


def allow_reattempt(student, exam_name, allowed_by="admin"):
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO re_attempts (student, exam_name, allowed, allowed_by, allowed_at)
               VALUES (?,?,1,?,?)""",
            (student, exam_name, allowed_by, int(time.time())),
        )


def check_reattempt(student, exam_name):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM re_attempts WHERE student=? AND exam_name=? AND allowed=1 AND used=0",
            (student, exam_name),
        ).fetchone()
        return dict(row) if row else None


def use_reattempt(student, exam_name):
    with get_db() as conn:
        conn.execute(
            "UPDATE re_attempts SET used=1, used_at=? WHERE student=? AND exam_name=?",
            (int(time.time()), student, exam_name),
        )


def list_reattempt_requests():
    with get_db() as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM re_attempts WHERE allowed=0 ORDER BY id DESC"
            ).fetchall()
        ]





def list_students_by_year():
    """Group all students by their academic year for the admin view."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM student_profiles ORDER BY year, full_name"
        ).fetchall()
        by_year = {}
        for r in rows:
            y = r["year"] or "Unassigned"
            if y not in by_year:
                by_year[y] = []
            by_year[y].append(dict(r))
        return by_year


def get_year_mates_results(student_username):
    """Get published results for everyone in the same year as the given student."""
    with get_db() as conn:
        profile = conn.execute(
            "SELECT year FROM student_profiles WHERE username=?", (student_username,)
        ).fetchone()
        if not profile or not profile["year"]:
            return []

        year = profile["year"]
        return [
            dict(r)
            for r in conn.execute(
                """SELECT r.*, sp.full_name, sp.student_id
               FROM results r
               JOIN student_profiles sp ON r.student = sp.username
               WHERE sp.year=? AND r.published=1
               ORDER BY r.marks DESC""",
                (year,),
            ).fetchall()
        ]


# ── Exam Settings Persistence ────────────────────────────────────────────────


def get_all_exam_settings():
    """List all saved exam settings."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM exam_settings ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_active_exam_students():
    """Returns students who currently have an active session (based on exam_sessions table)."""
    with get_db() as conn:
        # We consider a session active if it has been started but has no ended_at.
        # Note: Depending on implementation, you might need a separate 'active_exams' table.
        rows = conn.execute(
            """SELECT student as username, warnings, tab_switches, started_at, id as session_id
               FROM exam_sessions 
               WHERE ended_at IS NULL OR ended_at = 0 
               ORDER BY started_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
# ── Bulk Generating Exam Keys ──────────────────────────────────────────────────
import secrets
import string

def generate_bulk_keys(exam_name, count=10):
    """
    Generate multiple unique 8-character exam keys in a single transaction.
    This prevents the application from getting stuck during large batch creation.
    """
    new_keys = []
    with get_db() as conn:
        for _ in range(count):
            # Generate a 8-char random alphanumeric key
            key = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            try:
                conn.execute(
                    "INSERT INTO exam_keys (exam_name, key_value) VALUES (?, ?)",
                    (exam_name, key)
                )
                new_keys.append(key)
            except sqlite3.IntegrityError:
                # Key collision, skip and let the loop continue
                continue
    return new_keys

def list_exam_keys(exam_name):
    """List all keys (used and unused) for a specific exam."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM exam_keys WHERE exam_name=? ORDER BY created_at DESC", 
            (exam_name,)
        ).fetchall()
        return [dict(r) for r in rows]

def list_exams():
    """List all exam settings."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM exam_settings ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

def create_exam(exam_name, duration=60, total_marks=100):
    """Create a new exam setting."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO exam_settings (exam_name, duration, total_marks, is_active) 
               VALUES (?, ?, ?, 1)""",
            (exam_name, duration, total_marks)
        )
    return True
