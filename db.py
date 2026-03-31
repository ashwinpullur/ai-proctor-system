"""
db.py — Database module for ScoreHunt AI Proctor
Supports MySQL with fallback to SQLite if connection fails.
Handles: questions, exam sessions, events, evidence, hackathons, queries.
"""

import sqlite3
import pymysql
import os
import json
import time
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), 'proctor.db')

# MySQL Configuration (Default to localhost)
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'proctor_platform',
    'cursorclass': pymysql.cursors.DictCursor
}

db_mode = 'sqlite' # will switch to mysql if available

def check_mysql():
    global db_mode
    try:
        conn = pymysql.connect(host=MYSQL_CONFIG['host'], user=MYSQL_CONFIG['user'], password=MYSQL_CONFIG['password'])
        # Create database if not exists
        conn.cursor().execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
        conn.commit()
        conn.close()
        db_mode = 'mysql'
        print("[DB] Using MySQL backend.")
    except Exception as e:
        print(f"[DB] Failed to connect to MySQL ({e}). Falling back to SQLite.")
        db_mode = 'sqlite'

check_mysql()

class RowDict:
    def __init__(self, d):
        self.d = d
    def keys(self):
        return self.d.keys()
    def __getitem__(self, key):
        return self.d[key]

@contextmanager
def get_db():
    """Context manager: yields a connected database connection."""
    if db_mode == 'mysql':
        conn = pymysql.connect(**MYSQL_CONFIG)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

def _exec(conn, query, args=(), fetchall=False, fetchone=False):
    """Abstraction to handle both PyMySQL and SQLite syntax differences slightly."""
    is_sqlite = (db_mode == 'sqlite')
    if is_sqlite:
        # Replace %s with ? for SQLite
        query = query.replace('%s', '?')
    
    cur = conn.cursor()
    cur.execute(query, args)
    if fetchall:
        if is_sqlite:
            return [dict(r) for r in cur.fetchall()]
        else:
            return cur.fetchall()
        
    if fetchone:
        if is_sqlite:
            res = cur.fetchone()
            return dict(res) if res else None
        else:
            return cur.fetchone()
            
    return cur

def init_db():
    """Create all tables."""
    with get_db() as conn:
        if db_mode == 'sqlite':
            autoinc = "INTEGER PRIMARY KEY AUTOINCREMENT"
            # SQLite does not have json type, uses TEXT
            json_type = "TEXT"
        else:
            autoinc = "INT AUTO_INCREMENT PRIMARY KEY"
            json_type = "JSON"

        tables = f"""
            CREATE TABLE IF NOT EXISTS questions (
                id              {autoinc},
                type            VARCHAR(10) NOT NULL,
                question        TEXT NOT NULL,
                options         TEXT,
                correct_answer  INT,
                code_prompt     TEXT,
                placeholder     TEXT,
                created_at      INT NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS exam_sessions (
                id          VARCHAR(255) PRIMARY KEY,
                student     VARCHAR(255),
                started_at  INT,
                ended_at    INT,
                warnings    INT DEFAULT 0,
                tab_switches INT DEFAULT 0,
                verdict     VARCHAR(50) DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS session_events (
                id          {autoinc},
                session_id  VARCHAR(255) NOT NULL,
                timestamp   INT NOT NULL,
                msg         TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_evidence (
                id          {autoinc},
                session_id  VARCHAR(255) NOT NULL,
                filename    VARCHAR(255) NOT NULL,
                msg         TEXT,
                timestamp   INT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS students (
                username VARCHAR(255) PRIMARY KEY,
                student_id VARCHAR(100),
                year_category VARCHAR(50),
                marks INT DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS hackathons (
                id {autoinc},
                title VARCHAR(255) NOT NULL,
                link TEXT NOT NULL,
                description TEXT,
                created_at INT NOT NULL DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS queries (
                id {autoinc},
                student VARCHAR(255),
                message TEXT,
                admin_reply TEXT,
                timestamp INT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS exams_meta (
                id {autoinc},
                key_uploaded TINYINT(1) DEFAULT 0,
                results_released TINYINT(1) DEFAULT 0,
                duration_minutes INT DEFAULT 60
            );
        """
        if db_mode == 'sqlite':
            conn.executescript(tables)
        else:
            for statement in tables.split(';'):
                if statement.strip():
                    conn.cursor().execute(statement)
        
        # Insert default exam meta if not exists
        ex_meta = _exec(conn, "SELECT * FROM exams_meta", fetchone=True)
        if not ex_meta:
            _exec(conn, "INSERT INTO exams_meta (key_uploaded, results_released, duration_minutes) VALUES (0, 0, 60)")
            
        # Migrate schema for duration_minutes if missing
        try:
            _exec(conn, "ALTER TABLE exams_meta ADD COLUMN duration_minutes INTEGER DEFAULT 60")
        except Exception:
            pass # Already exists
            
        # Migrate schema for new student profile fields gracefully
        new_cols = [
            ("full_name", "VARCHAR(100)", "TEXT"),
            ("email", "VARCHAR(100)", "TEXT"),
            ("phone", "VARCHAR(50)", "TEXT"),
            ("gpa", "FLOAT", "REAL"),
            ("profile_image", "VARCHAR(255)", "TEXT"),
            ("resume", "VARCHAR(255)", "TEXT")
        ]
        
        for col, m_type, sq_type in new_cols:
            ctype = sq_type if db_mode == 'sqlite' else m_type
            try:
                _exec(conn, f"ALTER TABLE students ADD COLUMN {col} {ctype}")
            except Exception:
                pass # Already exists
            
    print(f"[DB] Database initialized -> {db_mode}")

# ── Feature CRUDs ─────────────────────────────────────────────────────────────

def get_exam_meta():
    with get_db() as conn:
        res = _exec(conn, "SELECT * FROM exams_meta LIMIT 1", fetchone=True)
        return res or {"key_uploaded": 0, "results_released": 0, "duration_minutes": 60}

def update_exam_meta(key_uploaded=None, results_released=None, duration_minutes=None):
    with get_db() as conn:
        meta = get_exam_meta()
        k = key_uploaded if key_uploaded is not None else meta.get('key_uploaded', 0)
        r = results_released if results_released is not None else meta.get('results_released', 0)
        d = duration_minutes if duration_minutes is not None else meta.get('duration_minutes', 60)
        _exec(conn, "UPDATE exams_meta SET key_uploaded=%s, results_released=%s, duration_minutes=%s", (k, r, d))

# ── Student Data ──────────────────────────────────────────────────────────────
def update_student(username, student_id=None, year_category=None, marks=None):
    with get_db() as conn:
        st = _exec(conn, "SELECT * FROM students WHERE username=%s", (username,), fetchone=True)
        if not st:
            _exec(conn, "INSERT INTO students (username, student_id, year_category, marks) VALUES (%s,%s,%s,%s)",
                  (username, student_id or '', year_category or '', marks or 0))
        else:
            sid = student_id if student_id is not None else st.get('student_id', '')
            yc = year_category if year_category is not None else st.get('year_category', '')
            m = marks if marks is not None else st.get('marks', 0)
            _exec(conn, "UPDATE students SET student_id=%s, year_category=%s, marks=%s WHERE username=%s",
                  (sid, yc, m, username))

def update_student_profile(username, full_name=None, email=None, phone=None, gpa=None, profile_image=None, resume=None):
    with get_db() as conn:
        st = _exec(conn, "SELECT * FROM students WHERE username=%s", (username,), fetchone=True)
        if not st:
            _exec(conn, "INSERT INTO students (username, full_name, email, phone, gpa, profile_image, resume) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                  (username, full_name, email, phone, gpa, profile_image, resume))
        else:
            # Only update given fields cleanly
            fields = []
            vals   = []
            for col, val in [("full_name", full_name), ("email", email), ("phone", phone), ("gpa", gpa), ("profile_image", profile_image), ("resume", resume)]:
                if val is not None:
                    fields.append(f"{col}=%s")
                    vals.append(val)
            if fields:
                vals.append(username)
                query = f"UPDATE students SET {', '.join(fields)} WHERE username=%s"
                _exec(conn, query, tuple(vals))

def get_student(username):
    with get_db() as conn:
        return _exec(conn, "SELECT * FROM students WHERE username=%s", (username,), fetchone=True)

def list_students():
    with get_db() as conn:
        return _exec(conn, "SELECT * FROM students", fetchall=True)

# ── Hackathons ────────────────────────────────────────────────────────────────
def add_hackathon(title, link, description):
    with get_db() as conn:
        _exec(conn, "INSERT INTO hackathons (title, link, description, created_at) VALUES (%s, %s, %s, %s)",
              (title, link, description, int(time.time())))

def list_hackathons():
    with get_db() as conn:
        return _exec(conn, "SELECT * FROM hackathons ORDER BY created_at DESC", fetchall=True)

def delete_hackathon(h_id):
    with get_db() as conn:
        _exec(conn, "DELETE FROM hackathons WHERE id=%s", (h_id,))

# ── Queries (Chat space) ──────────────────────────────────────────────────────
def add_query(student, message):
    with get_db() as conn:
        _exec(conn, "INSERT INTO queries (student, message, timestamp) VALUES (%s, %s, %s)",
              (student, message, int(time.time())))
              
def reply_query(q_id, admin_reply):
    with get_db() as conn:
        _exec(conn, "UPDATE queries SET admin_reply=%s WHERE id=%s", (admin_reply, q_id))

def get_student_queries(student):
    with get_db() as conn:
        return _exec(conn, "SELECT * FROM queries WHERE student=%s ORDER BY timestamp DESC", (student,), fetchall=True)

def list_all_queries():
    with get_db() as conn:
        return _exec(conn, "SELECT * FROM queries ORDER BY timestamp DESC", fetchall=True)


# ── Questions CRUD ─────────────────────────────────────────────────────────────

def add_question(q_type, question, options=None, correct_answer=None,
                  code_prompt=None, placeholder=None):
    opts_json = json.dumps(options) if options else None
    with get_db() as conn:
        cur = _exec(conn,
            """INSERT INTO questions
               (type, question, options, correct_answer, code_prompt, placeholder, created_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (q_type, question, opts_json, correct_answer, code_prompt, placeholder, int(time.time()))
        )
        return cur.lastrowid


def list_questions():
    """Return all questions as a list of dicts (options decoded from JSON)."""
    with get_db() as conn:
        rows = _exec(conn, "SELECT * FROM questions ORDER BY id", fetchall=True)
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
        cur = _exec(conn, "DELETE FROM questions WHERE id=%s", (q_id,))
        return cur.rowcount > 0


# ── Session CRUD ───────────────────────────────────────────────────────────────

def save_session(session_id, student, ended_at, started_at,
                 warnings, tab_switches, events, evidence):
    """Upsert a completed exam session and its events/evidence."""
    if warnings >= 15 or tab_switches >= 3:
        verdict = 'fail'
    elif warnings == 0:
        verdict = 'pass'
    else:
        verdict = 'review'

    with get_db() as conn:
        if db_mode == 'sqlite':
            _exec(conn,
                """INSERT OR REPLACE INTO exam_sessions
                   (id, student, started_at, ended_at, warnings, tab_switches, verdict)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (session_id, student, started_at, ended_at, warnings, tab_switches, verdict)
            )
        else:
            _exec(conn,
                """INSERT INTO exam_sessions
                   (id, student, started_at, ended_at, warnings, tab_switches, verdict)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   ON DUPLICATE KEY UPDATE student=VALUES(student), started_at=VALUES(started_at), ended_at=VALUES(ended_at), warnings=VALUES(warnings), tab_switches=VALUES(tab_switches), verdict=VALUES(verdict)""",
                (session_id, student, started_at, ended_at, warnings, tab_switches, verdict)
            )
            
        _exec(conn, "DELETE FROM session_events WHERE session_id=%s", (session_id,))
        _exec(conn, "DELETE FROM session_evidence WHERE session_id=%s", (session_id,))

        for ev in events:
            _exec(conn,
                "INSERT INTO session_events (session_id, timestamp, msg) VALUES (%s,%s,%s)",
                (session_id, ev.get('timestamp', int(time.time())), ev.get('msg', ''))
            )
        for ev in evidence:
            _exec(conn,
                """INSERT INTO session_evidence (session_id, filename, msg, timestamp)
                   VALUES (%s,%s,%s,%s)""",
                (session_id, ev.get('file', ''), ev.get('msg', ''),
                 ev.get('timestamp', int(time.time())))
            )


def list_sessions():
    """Return all sessions ordered newest-first."""
    with get_db() as conn:
        return _exec(conn,
            """SELECT es.*, (SELECT COUNT(id) FROM session_evidence se WHERE se.session_id = es.id) as evidence_count
               FROM exam_sessions es
               ORDER BY es.ended_at DESC""", fetchall=True
        )


def get_session(session_id):
    """Return a single session with its events and evidence."""
    with get_db() as conn:
        s = _exec(conn, "SELECT * FROM exam_sessions WHERE id=%s", (session_id,), fetchone=True)
        if not s:
            return None
        events = _exec(conn, "SELECT timestamp, msg FROM session_events WHERE session_id=%s ORDER BY timestamp", (session_id,), fetchall=True)
        evidence = _exec(conn, "SELECT filename as file, msg, timestamp FROM session_evidence WHERE session_id=%s", (session_id,), fetchall=True)

    result = dict(s)
    result['events']   = events
    result['evidence'] = evidence
    return result
def delete_student_session(username):
    """
    Remove session record and ALL evidence associated with a student to allow re-attempt.
    """
    with get_db() as conn:
        # 1. Delete evidence and events first by retrieving session IDs
        session_ids = _exec(conn, "SELECT id FROM exam_sessions WHERE student=%s", (username,), fetchall=True)
        for row in session_ids:
            sid = row['id']
            _exec(conn, "DELETE FROM session_evidence WHERE session_id=%s", (sid,))
            _exec(conn, "DELETE FROM session_events WHERE session_id=%s", (sid,))
        
        # 2. Delete session record
        _exec(conn, "DELETE FROM exam_sessions WHERE student=%s", (username,))
        # 3. Reset student marks in profile
        _exec(conn, "UPDATE students SET marks=0 WHERE username=%s", (username,))
    print(f"[DB] Session and evidence cleared for student: {username}")
    return True
