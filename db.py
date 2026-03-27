"""
db.py — SQLite database module for ScoreHunt AI Proctor
Handles: questions, exam sessions, events, and evidence storage.
"""

import sqlite3
import os
import json
import time
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), 'proctor.db')


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
                created_at      INTEGER NOT NULL DEFAULT (strftime('%s','now'))
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
        """)
    print("[DB] Database initialized →", DB_PATH)


# ── Questions CRUD ─────────────────────────────────────────────────────────────

def add_question(q_type, question, options=None, correct_answer=None,
                  code_prompt=None, placeholder=None):
    """Insert a new question. Returns the new row id."""
    opts_json = json.dumps(options) if options else None
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO questions
               (type, question, options, correct_answer, code_prompt, placeholder)
               VALUES (?,?,?,?,?,?)""",
            (q_type, question, opts_json, correct_answer, code_prompt, placeholder)
        )
        return cur.lastrowid


def list_questions():
    """Return all questions as a list of dicts (options decoded from JSON)."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM questions ORDER BY id").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d['options']:
            d['options'] = json.loads(d['options'])
        result.append(d)
    return result


def delete_question(q_id):
    """Delete a question by id. Returns True if deleted."""
    with get_db() as conn:
        cur = conn.execute("DELETE FROM questions WHERE id=?", (q_id,))
        return cur.rowcount > 0


# ── Session CRUD ───────────────────────────────────────────────────────────────

def save_session(session_id, student, ended_at, started_at,
                 warnings, tab_switches, events, evidence):
    """Upsert a completed exam session and its events/evidence."""
    # Determine verdict
    if warnings >= 15 or tab_switches >= 3:
        verdict = 'fail'
    elif warnings == 0:
        verdict = 'pass'
    else:
        verdict = 'review'

    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO exam_sessions
               (id, student, started_at, ended_at, warnings, tab_switches, verdict)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, student, started_at, ended_at, warnings, tab_switches, verdict)
        )
        # Delete old events/evidence for this session before reinserting
        conn.execute("DELETE FROM session_events WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM session_evidence WHERE session_id=?", (session_id,))

        for ev in events:
            conn.execute(
                "INSERT INTO session_events (session_id, timestamp, msg) VALUES (?,?,?)",
                (session_id, ev.get('timestamp', int(time.time())), ev.get('msg', ''))
            )
        for ev in evidence:
            conn.execute(
                """INSERT INTO session_evidence (session_id, filename, msg, timestamp)
                   VALUES (?,?,?,?)""",
                (session_id, ev.get('file', ''), ev.get('msg', ''),
                 ev.get('timestamp', int(time.time())))
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
            (session_id,)
        ).fetchall()
        evidence = conn.execute(
            "SELECT filename as file, msg, timestamp FROM session_evidence WHERE session_id=?",
            (session_id,)
        ).fetchall()

    result = dict(s)
    result['events']   = [dict(e) for e in events]
    result['evidence'] = [dict(e) for e in evidence]
    return result
