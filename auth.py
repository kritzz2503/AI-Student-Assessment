# auth.py
"""
Mock authentication for the 10 seeded demo accounts (5 teachers, 5 students).

SECURITY NOTE — read this before presenting this as "secure login" in a
viva: using a student's date of birth as their password is a well-known
WEAK practice in real systems, because a DOB is often public/guessable
information (school records, social media, ID cards). This is acceptable
ONLY because these are mock/demo accounts for a student project. A real
deployment would require:
  - User-chosen strong passwords (not DOB or any guessable personal data)
  - A password reset flow
  - Rate-limiting login attempts (to stop brute-force guessing)
  - Possibly multi-factor authentication for teacher/admin accounts

What this DOES do correctly: passwords are never stored in plain text.
They're hashed with Werkzeug's `generate_password_hash` (PBKDF2-based)
before being saved, and checked with `check_password_hash`, which is the
standard safe pattern — even though the underlying password choice (DOB)
is weak, at least the storage mechanism is not.
"""

import sqlite3
from werkzeug.security import check_password_hash


def verify_login(username, password):
    """
    Checks username + password against the users table.

    Returns a user dict on success:
        {
            "username": ...,
            "full_name": ...,
            "role": "teacher" | "student",
            "institution_name": ...
        }
    Returns None if the username doesn't exist or the password is wrong.
    Deliberately returns the same "invalid username or password" style
    result for both cases — never reveal whether it was the username or
    the password that was wrong, since that helps an attacker guess
    valid usernames.
    """
    if not username or not password:
        return None

    conn = sqlite3.connect("grading_system.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.username, u.password_hash, u.full_name, u.role, i.institution_name, u.roll_number
        FROM users u
        JOIN institutions i ON u.institution_id = i.institution_id
        WHERE u.username = ?
    """, (username,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    db_username, password_hash, full_name, role, institution_name, roll_number = row

    if not check_password_hash(password_hash, password):
        return None

    return {
        "username"         : db_username,
        "full_name"        : full_name,
        "role"             : role,
        "institution_name" : institution_name,
        "roll_number"      : roll_number or ""
    }
