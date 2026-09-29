# plagiarism.py
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from security import decrypt_value

PLAGIARISM_THRESHOLD = 0.92

# ── Inconsistency Detection Settings ───────────────────
# A z-score above this means an answer's writing style deviates
# sharply from the student's own baseline across the same exam.
INCONSISTENCY_Z_THRESHOLD = 2.5
MIN_ANSWERS_FOR_BASELINE  = 3   # need at least 3 other answers — fewer makes
                                 # the baseline's spread too noisy to trust
MIN_STD_FLOOR = {
    # If a metric's natural spread (std) across the student's own answers
    # is below this floor, treat it as "not enough variation to judge" and
    # skip that metric rather than risk a false positive from a tiny std.
    "avg_word_len"  : 0.5,
    "flesch_reading": 5.0,
    "gunning_fog"   : 1.5,
}


def check_plagiarism(student_name, exam_id, question_id, student_answer, bert_model):
    """
    Compare the student's answer against all previously submitted answers
    for the same question in the same exam.
    Returns (is_plagiarized: bool, flagged_matches: list)
    """
    if not student_answer or not student_answer.strip():
        return False, []

    conn = sqlite3.connect("grading_system.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT student_name, student_answer
        FROM submissions
        WHERE exam_id = ? AND question_id = ? AND student_name != ?
    """, (exam_id, question_id, student_name))
    others = cursor.fetchall()
    conn.close()

    if not others:
        return False, []

    student_emb = bert_model.encode([student_answer])
    flagged = []

    for other_name, other_answer_encrypted in others:
        other_answer = decrypt_value(other_answer_encrypted)
        if not other_answer or not other_answer.strip():
            continue
        other_emb = bert_model.encode([other_answer])
        sim = float(cosine_similarity(student_emb, other_emb)[0][0])
        if sim >= PLAGIARISM_THRESHOLD:
            flagged.append({
                "matched_student": other_name,
                "similarity_score": round(sim, 4)
            })

    return len(flagged) > 0, flagged


def _style_features(text):
    """
    Lightweight writing-style fingerprint for one answer.
    Uses textstat (already a project dependency) so no new install is needed.
    """
    import textstat
    text = str(text)
    words = text.split()
    if not words:
        return None
    return {
        "avg_word_len"  : float(np.mean([len(w) for w in words])),
        "word_count"    : len(words),
        "flesch_reading": float(textstat.flesch_reading_ease(text)),
        "gunning_fog"   : float(textstat.gunning_fog(text)),
    }


def check_inconsistency(student_name, exam_id, question_id, student_answer):
    """
    Flags an answer whose writing style deviates sharply from the SAME
    student's OTHER answers in the SAME exam (self-consistency check).

    This is distinct from plagiarism: plagiarism compares student-to-student,
    this compares a student against their own baseline. A big deviation can
    suggest an answer was written by someone else, copy-pasted from an
    external source, or produced under very different conditions than the
    rest of the exam.

    Returns (is_inconsistent: bool, detail: dict | None)
    """
    current_features = _style_features(student_answer)
    if current_features is None:
        return False, None

    conn = sqlite3.connect("grading_system.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT student_answer
        FROM submissions
        WHERE exam_id = ? AND student_name = ? AND question_id != ?
    """, (exam_id, student_name, question_id))
    raw_rows = [r[0] for r in cursor.fetchall()]
    conn.close()

    other_answers = [decrypt_value(a) for a in raw_rows]
    other_answers = [a for a in other_answers if a and a.strip()]

    if len(other_answers) < MIN_ANSWERS_FOR_BASELINE:
        # Not enough of the student's own history yet to judge "normal" style
        return False, None

    baseline_rows = [_style_features(a) for a in other_answers]
    baseline_rows = [r for r in baseline_rows if r is not None]
    if len(baseline_rows) < MIN_ANSWERS_FOR_BASELINE:
        return False, None

    deviations = {}
    flagged_metrics = []
    for metric in ["avg_word_len", "flesch_reading", "gunning_fog"]:
        baseline_vals = [r[metric] for r in baseline_rows]
        mean = float(np.mean(baseline_vals))
        std  = float(np.std(baseline_vals))
        floor = MIN_STD_FLOOR.get(metric, 1e-6)
        std = max(std, floor)  # avoid an artificially tiny std causing false alarms
        z = abs((current_features[metric] - mean) / std)
        deviations[metric] = round(z, 2)
        if z >= INCONSISTENCY_Z_THRESHOLD:
            flagged_metrics.append(metric)

    # Require at least 2 of the 3 style metrics to agree before flagging.
    # A single metric crossing threshold is weak evidence on its own;
    # multiple metrics agreeing is a much stronger signal.
    if len(flagged_metrics) >= 2:
        return True, {
            "flagged_metrics": flagged_metrics,
            "z_scores"       : deviations,
            "note"           : "This answer's writing style deviates noticeably "
                                "from this student's other answers in the same exam."
        }

    return False, None