# database.py
import sqlite3
from werkzeug.security import generate_password_hash

DEFAULT_INSTITUTION_NAME = "Default Institution"

# ── Mock login accounts ────────────────────────────────
# 5 teachers + 5 students, ALL in ONE institution, so every teacher can
# see the same shared exam and every student's submission, and so
# plagiarism checks have real same-question answers to compare across
# students. Password for every account = their DOB as DDMMYYYY (a
# simple, memorable scheme suitable for a mock/demo system — see
# SECURITY NOTE in auth.py for why this would NOT be acceptable in a
# real production system).
MOCK_USERS = [
    # (username, full_name, role, dob_ddmmyyyy, institution_name, roll_number)
    ("teacher1", "Mr. Arun Kumar",  "teacher", "15081985", "Greenfield High", "T001"),
    ("teacher2", "Ms. Divya Rao",   "teacher", "22031990", "Greenfield High", "T002"),
    ("teacher3", "Mr. Suresh Babu", "teacher", "09111982", "Greenfield High", "T003"),
    ("teacher4", "Ms. Priya Menon", "teacher", "30062001", "Greenfield High", "T004"),
    ("teacher5", "Mr. Anand Iyer",  "teacher", "17042002", "Greenfield High", "T005"),
    ("student1", "Jessi",           "student", "05012007", "Greenfield High", "GH2024001"),
    ("student2", "Karthik",         "student", "12092006", "Greenfield High", "GH2024002"),
    ("student3", "Meena",           "student", "28072007", "Greenfield High", "GH2024003"),
    ("student4", "Ravi",            "student", "03102006", "Greenfield High", "GH2024004"),
    ("student5", "Anjali",          "student", "19052007", "Greenfield High", "GH2024005"),
]


def init_db():
    conn = sqlite3.connect("grading_system.db")
    cursor = conn.cursor()

    # ── Multi-institution support (Scalability requirement) ──────
    # Each institution is data-isolated: every exam belongs to exactly one
    # institution, and questions/submissions inherit that isolation through
    # their exam_id. This lets one deployment of this system serve several
    # schools/colleges without their data ever mixing in a query.
    #
    # NOTE: this is application-level data isolation only. It does NOT by
    # itself make the system horizontally scalable (multiple servers, load
    # balancing) or cloud-deployed — that remains future work requiring
    # real infrastructure (e.g. a managed Postgres DB + container hosting),
    # which is outside what a code change alone can deliver.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS institutions (
            institution_id INTEGER PRIMARY KEY AUTOINCREMENT,
            institution_name TEXT NOT NULL UNIQUE
        )
    """)

    # Ensure a default institution exists so single-institution users
    # (and all pre-existing exams) have somewhere valid to belong to.
    cursor.execute(
        "INSERT OR IGNORE INTO institutions (institution_name) VALUES (?)",
        (DEFAULT_INSTITUTION_NAME,)
    )
    conn.commit()
    cursor.execute(
        "SELECT institution_id FROM institutions WHERE institution_name = ?",
        (DEFAULT_INSTITUTION_NAME,)
    )
    default_institution_id = cursor.fetchone()[0]

    # ── Login accounts (teachers + students) ──────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL,
            institution_id INTEGER NOT NULL,
            roll_number TEXT DEFAULT '',
            FOREIGN KEY (institution_id) REFERENCES institutions (institution_id)
        )
    """)

    # Add roll_number column if missing (older databases)
    cursor.execute("PRAGMA table_info(users)")
    user_cols = [row[1] for row in cursor.fetchall()]
    if "roll_number" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN roll_number TEXT DEFAULT ''")
        print("Added missing column: roll_number")

    # Seed the 10 mock accounts. MOCK_USERS above is treated as the
    # single source of truth for username/role/institution: if a user
    # already exists but their institution_id has drifted from what's
    # listed here (e.g. after editing this list), it gets corrected on
    # the next app startup. The password is only set ONCE, when a user
    # is first created — it's never silently overwritten afterwards,
    # since this method also runs in normal day-to-day startups.
    for username, full_name, role, dob, institution_name, roll_number in MOCK_USERS:
        cursor.execute(
            "INSERT OR IGNORE INTO institutions (institution_name) VALUES (?)",
            (institution_name,)
        )
        cursor.execute(
            "SELECT institution_id FROM institutions WHERE institution_name = ?",
            (institution_name,)
        )
        inst_id = cursor.fetchone()[0]

        cursor.execute("SELECT user_id, institution_id FROM users WHERE username = ?", (username,))
        existing = cursor.fetchone()

        if existing is None:
            password_hash = generate_password_hash(dob)
            cursor.execute("""
                INSERT INTO users (username, password_hash, full_name, role, institution_id, roll_number)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, password_hash, full_name, role, inst_id, roll_number))
        else:
            existing_user_id, existing_inst_id = existing
            if existing_inst_id != inst_id:
                cursor.execute(
                    "UPDATE users SET institution_id = ? WHERE user_id = ?",
                    (inst_id, existing_user_id)
                )
                print(f"Corrected institution for existing user '{username}' -> {institution_name}")
            # Update roll_number in case it was added later
            cursor.execute(
                "UPDATE users SET roll_number = ? WHERE user_id = ? AND (roll_number IS NULL OR roll_number = '')",
                (roll_number, existing_user_id)
            )

    # Stores each exam created by a teacher
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exams (
            exam_id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_name TEXT NOT NULL,
            institution_id INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (institution_id) REFERENCES institutions (institution_id)
        )
    """)

    # If exams already existed from before institution_id was added,
    # add the column now so older databases keep working without
    # deleting data. All existing exams are assigned to the default
    # institution so nothing becomes orphaned.
    cursor.execute("PRAGMA table_info(exams)")
    exam_cols = [row[1] for row in cursor.fetchall()]
    if "institution_id" not in exam_cols:
        cursor.execute(
            "ALTER TABLE exams ADD COLUMN institution_id INTEGER NOT NULL DEFAULT 1"
        )
        cursor.execute(
            "UPDATE exams SET institution_id = ? WHERE institution_id IS NULL",
            (default_institution_id,)
        )
        print("Added missing column: institution_id (existing exams assigned to default institution)")

    # ── One-time consolidation ──────────────────────────────────
    # Earlier in this project's setup, mock users/exams were briefly
    # split across "Greenfield High" and "Sunrise College" for a
    # multi-institution demo. Per a later decision, ALL 10 mock users
    # and their exams should share a single institution instead. This
    # moves any leftover exams from "Sunrise College" into
    # "Greenfield High" so nothing is silently invisible to a teacher
    # after this change. Safe to run repeatedly — once nothing is left
    # under the old name, this becomes a no-op.
    cursor.execute("SELECT institution_id FROM institutions WHERE institution_name = 'Sunrise College'")
    old_row = cursor.fetchone()
    cursor.execute("SELECT institution_id FROM institutions WHERE institution_name = 'Greenfield High'")
    gf_row = cursor.fetchone()
    if old_row and gf_row:
        old_id, gf_id = old_row[0], gf_row[0]
        cursor.execute("UPDATE exams SET institution_id = ? WHERE institution_id = ?", (gf_id, old_id))
        moved = cursor.rowcount
        if moved:
            print(f"Moved {moved} exam(s) from 'Sunrise College' into 'Greenfield High'")

    # Stores every question in an exam, with type and correct answer/model answer
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            question_id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_id INTEGER,
            question_number INTEGER,
            question_text TEXT,
            question_type TEXT,         -- 'mcq', 'short_answer', 'essay'
            correct_answer TEXT,        -- for MCQ: 'A'/'B'/'C'/'D'
            model_answer TEXT,          -- for short_answer: ideal answer text
            max_marks REAL,
            FOREIGN KEY (exam_id) REFERENCES exams (exam_id)
        )
    """)

    # Stores each student's submitted answer per question, plus the grade
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_id INTEGER,
            student_name TEXT,
            question_id INTEGER,
            student_answer TEXT,
            awarded_marks REAL,
            feedback TEXT,
            is_plagiarized INTEGER,
            is_inconsistent INTEGER DEFAULT 0,
            FOREIGN KEY (exam_id) REFERENCES exams (exam_id),
            FOREIGN KEY (question_id) REFERENCES questions (question_id)
        )
    """)

    # If the table already existed from before this column was added,
    # add it now so older databases keep working without deleting data.
    cursor.execute("PRAGMA table_info(submissions)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if "is_inconsistent" not in existing_cols:
        cursor.execute("ALTER TABLE submissions ADD COLUMN is_inconsistent INTEGER DEFAULT 0")
        print("Added missing column: is_inconsistent")

    conn.commit()
    conn.close()
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()