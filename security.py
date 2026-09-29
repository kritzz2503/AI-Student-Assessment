# security.py
"""
Column-level encryption for sensitive data (student names, student answers)
stored in SQLite.

Why column-level instead of full-file (SQLCipher) encryption:
- SQLCipher needs a separate compiled driver (pysqlcipher3 / sqlcipher3-binary)
  that isn't always available or simple to install cleanly on every machine,
  especially on Windows.
- Python's `cryptography` library is pure-Python-friendly, well-maintained,
  and works the same everywhere — fewer moving parts, easier to demo and explain.

This is NOT a substitute for a full security audit. It protects the data
*at rest* (if someone copies the .db file, they can't read names/answers
without the key). It does not protect data in transit (use HTTPS for that
in a real deployment) and the key itself must be kept safe — see notes below.
"""

import os
from cryptography.fernet import Fernet, InvalidToken

KEY_FILE = "secret.key"


def load_or_create_key():
    """
    Loads the encryption key from secret.key, or creates one if missing.

    IMPORTANT: secret.key must NEVER be committed to git or zipped into a
    submission, for the same reason as .env — anyone with this file can
    decrypt all stored student names and answers. Add it to .gitignore.
    """
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print(f"[security] New encryption key created at {KEY_FILE}. "
          f"Keep this file private and back it up — losing it makes "
          f"existing encrypted data unrecoverable.")
    return key


_key = load_or_create_key()
_fernet = Fernet(_key)


def encrypt_value(plain_text):
    """Encrypts a string. Returns the encrypted string (safe to store as TEXT)."""
    if plain_text is None:
        return None
    return _fernet.encrypt(str(plain_text).encode("utf-8")).decode("utf-8")


def decrypt_value(stored_value):
    """
    Decrypts a value that was encrypted with encrypt_value().

    Tolerant of OLD plain-text data that existed before encryption was
    added — if decryption fails (because the value was never encrypted),
    it just returns the original value unchanged instead of crashing.
    This lets old and new rows coexist safely in the same database.
    """
    if stored_value is None:
        return None
    try:
        return _fernet.decrypt(stored_value.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError, UnicodeDecodeError):
        # Not encrypted (old data) — return as-is
        return stored_value
