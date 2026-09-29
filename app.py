# app.py
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import sqlite3
import pickle
import json
import os
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from dotenv import load_dotenv
from groq import Groq
from extractor import extract_text_from_file, parse_response_sheet
from grader_router import route_and_grade
from plagiarism import check_plagiarism, check_inconsistency
from database import init_db
from security import encrypt_value, decrypt_value
from auth import verify_login

# ── Ensure DB schema is up to date (adds new columns if missing) ──
init_db()

# ── Load environment variables (API key from .env) ─────
load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── JSON Encoder to handle numpy types ────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Groq Client (key loaded safely from .env) ─────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Load trained models ────────────────────────────────
with open("ridge_model.pkl", "rb") as f:
    ridge_model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

from sentence_transformers import SentenceTransformer
import textstat

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

print("All models loaded successfully!")
print(f"Ridge model coefficients : {ridge_model.coef_.shape}")
print(f"Vectorizer vocabulary    : {len(vectorizer.vocabulary_)}")


# ── Feature Extraction (9 features — matches training) ─
def extract_features(text):
    try:
        text          = str(text)[:5000]
        words         = text.split()
        sentences     = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

        word_count        = len(words)
        sentence_count    = len(sentences) if len(sentences) > 0 else 1
        avg_word_len      = float(np.mean([len(w) for w in words])) if words else 0.0
        avg_sent_len      = float(word_count / sentence_count)
        unique_words      = len(set(words))
        lexical_diversity = float(unique_words / word_count) if word_count > 0 else 0.0
        flesch_reading    = float(textstat.flesch_reading_ease(text))
        flesch_kincaid    = float(textstat.flesch_kincaid_grade(text))
        gunning_fog       = float(textstat.gunning_fog(text))

        return [
            word_count, sentence_count, avg_word_len, avg_sent_len,
            unique_words, lexical_diversity,
            flesch_reading, flesch_kincaid, gunning_fog
        ]
    except:
        return [0.0] * 9


# ── Essay Score Prediction ─────────────────────────────
def predict_essay_score(new_essay):
    cleaned = new_essay.lower()
    cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    new_tfidf           = vectorizer.transform([cleaned])
    new_features_sparse = csr_matrix([extract_features(cleaned)])
    new_bert_sparse     = csr_matrix(bert_model.encode([cleaned]))
    new_combined        = hstack([new_tfidf, new_features_sparse, new_bert_sparse])

    return round(float(ridge_model.predict(new_combined)[0]), 2)


# ── Feedback Generation ────────────────────────────────
def generate_feedback(essay_text, predicted_score, max_score=6):
    score_percent = (predicted_score / max_score) * 100

    if score_percent >= 85:
        level = "Excellent"
    elif score_percent >= 70:
        level = "Good"
    elif score_percent >= 50:
        level = "Average"
    else:
        level = "Needs Improvement"

    prompt = f"""You are an expert essay evaluator.
Student essay score: {predicted_score:.1f}/{max_score} ({level}).
Essay: \"\"\"{essay_text[:800]}\"\"\"
Respond ONLY in this JSON format, no extra text:
{{
  "strengths": ["point1", "point2"],
  "weaknesses": ["point1", "point2"],
  "suggestions": ["point1", "point2"],
  "overall_comment": "one encouraging sentence"
}}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        fb  = json.loads(raw)
        fb["feedback_source"] = "ai"
    except Exception as e:
        import traceback
        print(f"[generate_feedback] AI feedback failed: {e}")
        print(traceback.format_exc())
        fb = {
            "strengths"       : ["Shows understanding of topic"],
            "weaknesses"      : ["Needs more detail"],
            "suggestions"     : ["Add examples", "Expand arguments"],
            "overall_comment" : "Keep working hard, you will improve!",
            "feedback_source" : "fallback",
            "fallback_reason" : str(e)
        }

    fb["level"]           = level
    fb["predicted_score"] = float(predicted_score)
    fb["max_score"]       = float(max_score)
    return fb


# ── ROUTE 1: Health Check ──────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})


# ── ROUTE 1b: Login ─────────────────────────────────────
@app.route("/login", methods=["POST"])
def login():
    """
    Mock login for the 10 seeded demo accounts (5 teachers, 5 students).
    See auth.py for the SECURITY NOTE on why DOB-as-password is only
    acceptable for this demo, not for a real deployment.
    """
    try:
        data     = request.get_json()
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()

        user = verify_login(username, password)
        if user is None:
            return jsonify({"error": "Invalid username or password"}), 401

        return jsonify({"status": "success", "user": user})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500


# ── ROUTE 2: Grade Essay Only ──────────────────────────
@app.route("/grade_essay", methods=["POST"])
def grade_essay():
    try:
        data       = request.get_json()
        essay_text = data.get("essay", "")
        if not essay_text.strip():
            return jsonify({"error": "Essay is empty"}), 400
        predicted = predict_essay_score(essay_text)
        feedback  = generate_feedback(essay_text, predicted)
        return jsonify({
            "predicted_score" : predicted,
            "feedback"        : feedback,
            "status"          : "success"
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ── ROUTE 3: Teacher Creates Exam ─────────────────────
@app.route("/create_exam", methods=["POST"])
def create_exam():
    try:
        data      = request.get_json()
        exam_name = data["exam_name"]
        questions = data["questions"]
        # Optional: which institution this exam belongs to. If the frontend
        # doesn't send this (e.g. existing index.html doesn't yet), it falls
        # back to "Default Institution" so nothing breaks for single-school use.
        institution_name = data.get("institution_name", "Default Institution")

        conn   = sqlite3.connect("grading_system.db")
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR IGNORE INTO institutions (institution_name) VALUES (?)",
            (institution_name,)
        )
        cursor.execute(
            "SELECT institution_id FROM institutions WHERE institution_name = ?",
            (institution_name,)
        )
        institution_id = cursor.fetchone()[0]

        cursor.execute(
            "INSERT INTO exams (exam_name, institution_id) VALUES (?, ?)",
            (exam_name, institution_id)
        )
        exam_id = cursor.lastrowid

        for q in questions:
            cursor.execute("""
                INSERT INTO questions
                (exam_id, question_number, question_text, question_type,
                 correct_answer, model_answer, max_marks)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                exam_id,
                q["question_number"],
                q.get("question_text", ""),
                q["question_type"],
                q.get("correct_answer", ""),
                q.get("model_answer", ""),
                q["max_marks"]
            ))

        conn.commit()
        conn.close()
        return jsonify({
            "status"          : "success",
            "exam_id"         : exam_id,
            "institution_name": institution_name
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ── ROUTE 4: Student Submits Response Sheet ────────────
@app.route("/submit_response", methods=["POST"])
def submit_response():
    try:
        file         = request.files["response_file"]
        exam_id      = request.form["exam_id"]
        student_name = request.form["student_name"]

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print(f"[1] File saved: {filepath}")

        raw_text        = extract_text_from_file(filepath)
        print(f"[2] Extracted text: {raw_text[:200]}")

        student_answers = parse_response_sheet(raw_text)
        print(f"[3] Parsed answers: {student_answers}")

        conn   = sqlite3.connect("grading_system.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM questions WHERE exam_id = ?", (exam_id,))
        questions = cursor.fetchall()
        print(f"[4] Questions in DB: {len(questions)}")

        # A student re-submitting the same exam replaces their earlier attempt
        cursor.execute("DELETE FROM submissions WHERE exam_id = ? AND student_name = ?",
                       (exam_id, student_name))
        conn.commit()

        results       = []
        total_awarded = 0.0
        total_max     = 0.0

        for q in questions:
            q_id, _, q_num, q_text, q_type, correct_ans, model_ans, max_marks = q
            student_ans = student_answers.get(q_num, "")
            print(f"[5] Q{q_num} ({q_type}) → {student_ans[:60]}")

            question_data = {
                "correct_answer" : correct_ans,
                "model_answer"   : model_ans,
                "max_marks"      : float(max_marks),
                "question_text"  : q_text
            }

            if not student_ans.strip():
                grade_result = {"awarded_marks": 0.0, "note": "No answer found"}
            else:
                grade_result = route_and_grade(
                    q_type, student_ans, question_data,
                    bert_model=bert_model,
                    predict_essay_score_fn=predict_essay_score,
                    groq_client=groq_client
                )
            print(f"[6] Grade result: {grade_result}")

            # ── Plagiarism Check ───────────────────────
            is_plagiarized = 0
            plagiarism_matches = []
            if student_ans.strip() and q_type in ["short_answer", "essay"]:
                plagiarized, matches = check_plagiarism(
                    student_name, exam_id, q_id, student_ans, bert_model
                )
                is_plagiarized     = int(plagiarized)
                plagiarism_matches = matches
                if plagiarized:
                    print(f"[!] PLAGIARISM DETECTED for Q{q_num}: {matches}")

            # ── Inconsistency Check (self-consistency, not vs other students) ──
            is_inconsistent = 0
            inconsistency_detail = None
            if student_ans.strip() and q_type in ["short_answer", "essay"]:
                inconsistent, detail = check_inconsistency(
                    student_name, exam_id, q_id, student_ans
                )
                is_inconsistent       = int(inconsistent)
                inconsistency_detail  = detail
                if inconsistent:
                    print(f"[!] INCONSISTENCY DETECTED for Q{q_num}: {detail}")

            feedback_data = {}
            if q_type == "essay" and student_ans.strip():
                feedback_data = generate_feedback(
                    student_ans, grade_result["awarded_marks"], max_score=float(max_marks)
                )
                print(f"[7] Feedback generated ✅")

            total_awarded += float(grade_result["awarded_marks"])
            total_max     += float(max_marks)

            # Encrypt the answer text only at the point of storage.
            # All grading, plagiarism, and inconsistency checks above this
            # line use the PLAIN student_ans — encryption must happen last,
            # otherwise the AI models would be comparing ciphertext.
            encrypted_answer = encrypt_value(student_ans)

            cursor.execute("""
                INSERT INTO submissions
                (exam_id, student_name, question_id, student_answer,
                 awarded_marks, feedback, is_plagiarized, is_inconsistent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exam_id, student_name, q_id, encrypted_answer,
                float(grade_result["awarded_marks"]),
                json.dumps({**grade_result, **feedback_data}, cls=NumpyEncoder),
                is_plagiarized,
                is_inconsistent
            ))

            results.append({
                "question_number"      : int(q_num),
                "question_type"        : q_type,
                "student_answer"       : student_ans,
                "max_marks"            : float(max_marks),
                "feedback"             : feedback_data,
                "is_plagiarized"       : bool(is_plagiarized),
                "plagiarism_matches"   : plagiarism_matches,
                "is_inconsistent"      : bool(is_inconsistent),
                "inconsistency_detail" : inconsistency_detail,
                **{k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in grade_result.items()}
            })

        conn.commit()
        conn.close()
        print(f"[8] ✅ Done! Total: {total_awarded}/{total_max}")

        response_data = {
            "student_name"     : student_name,
            "total_awarded"    : round(total_awarded, 2),
            "total_max"        : total_max,
            "percentage"       : round((total_awarded / total_max) * 100, 2) if total_max > 0 else 0,
            "question_results" : results
        }
        return Response(
            json.dumps(response_data, cls=NumpyEncoder),
            mimetype="application/json"
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("FULL ERROR:")
        print(error_details)
        return jsonify({"error": str(e), "details": error_details}), 500


# ── Shared Question Generation Logic ───────────────────
def _generate_questions_via_ai(topic, level, count, q_type):
    """
    Core AI question-generation call, shared by:
    - ROUTE 5 (/generate_questions): teacher manually picks the level
    - ROUTE 7 (/next_question): system picks the level automatically
      based on the student's running performance (adaptive difficulty)
    """
    prompt = f"""You are an expert educator. Generate {count} {q_type} questions on the topic: "{topic}".
Student competency level: {level}.
Respond ONLY in this JSON format with no extra text or markdown:
{{
  "questions": [
    {{
      "question_text": "Full question text here. For MCQ include options like: A) ... B) ... C) ... D) ...",
      "question_type": "{q_type}",
      "correct_answer": "A",
      "model_answer": "Ideal answer text here",
      "max_marks": 5
    }}
  ]
}}
Rules:
- For mcq: correct_answer must be A, B, C, or D. Include 4 labelled options inside question_text. model_answer can be empty.
- For short_answer: correct_answer is empty. model_answer is the ideal 1-2 sentence answer.
- For essay: both correct_answer and model_answer are empty strings.
- Adjust difficulty based on level: easy=basic recall, medium=understanding, hard=analysis/application."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.8
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    return result["questions"]


def compute_adaptive_level(student_name, exam_id):
    """
    Looks at the student's submissions so far in this exam and decides
    the next question's difficulty based on running average percentage:
        >= 75%  -> hard    (doing well, raise the bar)
        40-75%  -> medium  (steady, keep pace)
        < 40%   -> easy    (struggling, ease up)
    Returns (level: str, avg_percent: float | None, questions_seen: int)
    """
    conn = sqlite3.connect("grading_system.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.awarded_marks, q.max_marks
        FROM submissions s
        JOIN questions q ON s.question_id = q.question_id
        WHERE s.exam_id = ? AND s.student_name = ?
    """, (exam_id, student_name))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        # No history yet for this student in this exam — start at medium
        return "medium", None, 0

    total_awarded = sum(r[0] for r in rows)
    total_max     = sum(r[1] for r in rows)
    avg_percent   = (total_awarded / total_max) * 100 if total_max > 0 else 0

    if avg_percent >= 75:
        level = "hard"
    elif avg_percent >= 40:
        level = "medium"
    else:
        level = "easy"

    return level, round(avg_percent, 2), len(rows)


# ── ROUTE 5: AI Question Generator ────────────────────
@app.route("/generate_questions", methods=["POST"])
def generate_questions():
    try:
        data   = request.get_json()
        topic  = data.get("topic", "General Knowledge")
        level  = data.get("level", "medium")          # easy / medium / hard
        count  = int(data.get("count", 3))
        q_type = data.get("question_type", "mcq")     # mcq / short_answer / essay

        questions = _generate_questions_via_ai(topic, level, count, q_type)
        return jsonify({"status": "success", "questions": questions})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500


# ── ROUTE 7: Adaptive Next Question ───────────────────
@app.route("/next_question", methods=["POST"])
def next_question():
    """
    Adaptive assessment: instead of a teacher manually choosing the
    difficulty, this route looks at the student's running performance
    in the current exam and automatically picks the next question's
    difficulty level (easy / medium / hard), then generates it.
    """
    try:
        data         = request.get_json()
        student_name = data["student_name"]
        exam_id      = data["exam_id"]
        topic        = data.get("topic", "General Knowledge")
        q_type       = data.get("question_type", "mcq")
        count        = int(data.get("count", 1))

        level, avg_percent, questions_seen = compute_adaptive_level(student_name, exam_id)
        questions = _generate_questions_via_ai(topic, level, count, q_type)

        return jsonify({
            "status"          : "success",
            "chosen_level"    : level,
            "student_avg_pct" : avg_percent,
            "questions_seen"  : questions_seen,
            "questions"       : questions
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500


# ── ROUTE 8: List Exams (optionally scoped by institution) ────
@app.route("/list_exams", methods=["GET"])
def list_exams():
    """
    Returns exams, optionally filtered to one institution.
    Used by the frontend to populate the Exam ID dropdown so a logged-in
    teacher/student only sees exams belonging to their own institution.
    """
    try:
        institution_name = request.args.get("institution_name")

        conn   = sqlite3.connect("grading_system.db")
        cursor = conn.cursor()

        if institution_name:
            cursor.execute("""
                SELECT e.exam_id, e.exam_name
                FROM exams e
                JOIN institutions i ON e.institution_id = i.institution_id
                WHERE i.institution_name = ?
                ORDER BY e.exam_id DESC
            """, (institution_name,))
        else:
            cursor.execute("SELECT exam_id, exam_name FROM exams ORDER BY exam_id DESC")

        exams = [{"exam_id": r[0], "exam_name": r[1]} for r in cursor.fetchall()]
        conn.close()
        return jsonify({"status": "success", "exams": exams})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500


# ── ROUTE 6: Analytics Dashboard Data ─────────────────
@app.route("/analytics", methods=["GET"])
def analytics():
    try:
        # Optional institution filter (Scalability: multi-institution isolation).
        # Example: GET /analytics?institution_name=Greenfield%20High
        # Omit it to see all institutions combined (useful for a single-school setup).
        institution_name = request.args.get("institution_name")

        conn   = sqlite3.connect("grading_system.db")
        cursor = conn.cursor()

        institution_filter_sql = ""
        params = []
        if institution_name:
            institution_filter_sql = """
                AND e.exam_id IN (
                    SELECT exam_id FROM exams ex
                    JOIN institutions i ON ex.institution_id = i.institution_id
                    WHERE i.institution_name = ?
                )
            """
            params.append(institution_name)

        # Per-exam: average score and student count
        cursor.execute(f"""
            SELECT e.exam_name,
                   ROUND(AVG(s.awarded_marks), 2),
                   COUNT(DISTINCT s.student_name),
                   SUM(CASE WHEN s.is_plagiarized = 1 THEN 1 ELSE 0 END)
            FROM submissions s
            JOIN exams e ON s.exam_id = e.exam_id
            WHERE 1=1 {institution_filter_sql}
            GROUP BY s.exam_id
        """, params)
        exam_stats = [
            {
                "exam"             : r[0],
                "avg_score"        : r[1],
                "student_count"    : r[2],
                "plagiarism_flags" : r[3]
            }
            for r in cursor.fetchall()
        ]

        # Per-student: total marks, number of questions attempted, plagiarism count
        cursor.execute(f"""
            SELECT s.student_name,
                   ROUND(SUM(s.awarded_marks), 2),
                   COUNT(*),
                   SUM(CASE WHEN s.is_plagiarized = 1 THEN 1 ELSE 0 END),
                   ROUND(SUM(q.max_marks), 2)
            FROM submissions s
            JOIN exams e ON s.exam_id = e.exam_id
            JOIN questions q ON s.question_id = q.question_id
            WHERE 1=1 {institution_filter_sql}
            GROUP BY s.student_name
            ORDER BY SUM(s.awarded_marks) DESC
        """, params)
        student_stats = [
            {
                "student"          : r[0],
                "total_marks"      : r[1],
                "questions_done"   : r[2],
                "plagiarism_flags" : r[3],
                "total_max_marks"  : r[4]
            }
            for r in cursor.fetchall()
        ]

        # Score distribution across all submissions
        cursor.execute(f"""
            SELECT s.awarded_marks
            FROM submissions s
            JOIN exams e ON s.exam_id = e.exam_id
            WHERE 1=1 {institution_filter_sql}
        """, params)
        all_scores = [row[0] for row in cursor.fetchall()]

        # Question type breakdown
        cursor.execute(f"""
            SELECT q.question_type, COUNT(*), ROUND(AVG(s.awarded_marks), 2)
            FROM submissions s
            JOIN questions q ON s.question_id = q.question_id
            JOIN exams e ON s.exam_id = e.exam_id
            WHERE 1=1 {institution_filter_sql}
            GROUP BY q.question_type
        """, params)
        type_breakdown = [
            {"type": r[0], "count": r[1], "avg_marks": r[2]}
            for r in cursor.fetchall()
        ]

        # Plagiarism summary
        cursor.execute(f"""
            SELECT s.student_name, COUNT(*) as flags
            FROM submissions s
            JOIN exams e ON s.exam_id = e.exam_id
            WHERE s.is_plagiarized = 1 {institution_filter_sql}
            GROUP BY s.student_name ORDER BY flags DESC
        """, params)
        plagiarism_summary = [
            {"student": r[0], "flag_count": r[1]}
            for r in cursor.fetchall()
        ]

        conn.close()

        return jsonify({
            "institution_filter": institution_name,
            "exam_stats"        : exam_stats,
            "student_stats"     : student_stats,
            "all_scores"        : all_scores,
            "type_breakdown"    : type_breakdown,
            "plagiarism_summary": plagiarism_summary
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500


# ── PDF Report Generation ──────────────────────────────
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from flask import send_file


def _build_report_pdf(student_name, exam_id, roll_number=None):
    """
    Builds a PDF report for one student's exam submission.
    Returns a BytesIO buffer ready to send as a file download.
    PDF name: <StudentName>_<ExamName>_Report.pdf
    """
    conn   = sqlite3.connect("grading_system.db")
    cursor = conn.cursor()

    # Exam name
    cursor.execute("SELECT exam_name FROM exams WHERE exam_id = ?", (exam_id,))
    row = cursor.fetchone()
    exam_name = row[0] if row else f"Exam {exam_id}"

    # Student roll number from DB if not passed
    if not roll_number:
        cursor.execute("SELECT roll_number FROM users WHERE full_name = ?", (student_name,))
        r = cursor.fetchone()
        roll_number = r[0] if r else "—"

    # All submissions for this student + exam
    cursor.execute("""
        SELECT q.question_number, q.question_text, q.question_type,
               q.correct_answer, q.model_answer, q.max_marks,
               s.student_answer, s.awarded_marks, s.feedback,
               s.is_plagiarized, s.is_inconsistent
        FROM submissions s
        JOIN questions q ON s.question_id = q.question_id
        WHERE s.exam_id = ? AND s.student_name = ?
        ORDER BY q.question_number
    """, (exam_id, student_name))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None, exam_name

    total_awarded = sum(r[7] for r in rows)
    total_max     = sum(r[5] for r in rows)
    percentage    = round((total_awarded / total_max) * 100, 1) if total_max > 0 else 0

    # Competency label
    if percentage >= 85:
        competency = "Excellent"
    elif percentage >= 70:
        competency = "Good"
    elif percentage >= 50:
        competency = "Average"
    else:
        competency = "Needs Improvement"

    # ── Build PDF ──────────────────────────────────────
    buffer = BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#1e3a5f"),
        spaceAfter=4, alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        "SubTitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER, spaceAfter=2
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=11, textColor=colors.HexColor("#1e3a5f"),
        spaceBefore=12, spaceAfter=4,
        borderPad=4
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#333333"),
        spaceAfter=3, leading=14
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#777777"),
        spaceAfter=1
    )
    flag_style = ParagraphStyle(
        "Flag", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#cc2200"),
        spaceAfter=2
    )
    fb_style = ParagraphStyle(
        "FB", parent=styles["Normal"],
        fontSize=8.5, textColor=colors.HexColor("#1a4a6b"),
        leftIndent=10, spaceAfter=2, leading=13
    )

    story = []

    # ── Header ────────────────────────────────────────
    story.append(Paragraph("AI Grading System", title_style))
    story.append(Paragraph("Student Assessment Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1e3a5f")))
    story.append(Spacer(1, 10))

    # ── Student Info Table ─────────────────────────────
    info_data = [
        ["Student Name", student_name,   "Roll Number", roll_number or "—"],
        ["Exam",         exam_name,       "Exam ID",     str(exam_id)],
        ["Total Marks",  f"{total_awarded} / {total_max}", "Percentage", f"{percentage}%"],
        ["Competency",   competency,      "",            ""],
    ]
    info_table = Table(info_data, colWidths=[3.5*cm, 6*cm, 3.5*cm, 4*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#e8f0fe")),
        ("BACKGROUND",  (2, 0), (2, -1), colors.HexColor("#e8f0fe")),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 0), (-1, -1), colors.HexColor("#333333")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f5f8ff"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING",     (0, 0), (-1, -1), 6),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 14))

    # ── Per Question Results ───────────────────────────
    story.append(Paragraph("Question-wise Results", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
    story.append(Spacer(1, 6))

    for (q_num, q_text, q_type, correct_ans, model_ans, max_marks,
         student_ans_enc, awarded, feedback_json, is_plag, is_incon) in rows:

        # Decrypt student answer
        try:
            student_ans = decrypt_value(student_ans_enc) if student_ans_enc else "—"
        except Exception:
            student_ans = "[encrypted]"

        # Question header row
        q_label = f"Q{q_num}  |  {q_type.replace('_', ' ').title()}  |  Marks: {awarded} / {max_marks}"
        story.append(Paragraph(q_label, section_style))

        # Question text
        story.append(Paragraph(f"<b>Question:</b> {q_text or '—'}", body_style))

        # Student answer
        story.append(Paragraph("<b>Student's Answer:</b>", label_style))
        story.append(Paragraph(str(student_ans)[:1000] or "No answer", body_style))

        # Correct / Model answer
        if q_type == "mcq" and correct_ans:
            story.append(Paragraph(f"<b>Correct Answer:</b> {correct_ans}", body_style))
        elif model_ans:
            story.append(Paragraph("<b>Model Answer:</b>", label_style))
            story.append(Paragraph(str(model_ans)[:600], body_style))

        # Flags
        if is_plag:
            story.append(Paragraph("⚠ Plagiarism Detected", flag_style))
        if is_incon:
            story.append(Paragraph("⚠ Writing Inconsistency Detected", flag_style))

        # Essay Feedback
        if feedback_json:
            try:
                fb = json.loads(feedback_json) if isinstance(feedback_json, str) else feedback_json
                if fb.get("strengths"):
                    story.append(Paragraph("<b>AI Feedback:</b>", label_style))
                    if fb.get("strengths"):
                        story.append(Paragraph("Strengths:", label_style))
                        for pt in fb["strengths"]:
                            story.append(Paragraph(f"• {pt}", fb_style))
                    if fb.get("weaknesses"):
                        story.append(Paragraph("Weaknesses:", label_style))
                        for pt in fb["weaknesses"]:
                            story.append(Paragraph(f"• {pt}", fb_style))
                    if fb.get("suggestions"):
                        story.append(Paragraph("Suggestions:", label_style))
                        for pt in fb["suggestions"]:
                            story.append(Paragraph(f"• {pt}", fb_style))
                    if fb.get("overall_comment"):
                        story.append(Paragraph(
                            f"<i>{fb['overall_comment']}</i>", fb_style
                        ))
            except Exception:
                pass

        story.append(HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#dddddd")))
        story.append(Spacer(1, 6))

    # ── Footer summary ─────────────────────────────────
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a5f")))
    footer_data = [[
        f"Total: {total_awarded} / {total_max}",
        f"Percentage: {percentage}%",
        f"Competency: {competency}"
    ]]
    footer_table = Table(footer_data, colWidths=[6*cm, 5*cm, 6*cm])
    footer_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",   (0, 0), (-1, -1), colors.white),
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("PADDING",     (0, 0), (-1, -1), 8),
    ]))
    story.append(footer_table)

    doc.build(story)
    buffer.seek(0)
    return buffer, exam_name


# ── ROUTE: Student downloads their own report ──────────
@app.route("/download_report", methods=["GET"])
def download_report():
    """
    GET /download_report?student_name=Jessi&exam_id=1&roll_number=GH2024001
    Returns a PDF named: Jessi_MidtermExam_Report.pdf
    Student can download their own; teacher can download anyone's.
    """
    try:
        student_name = request.args.get("student_name", "").strip()
        exam_id      = request.args.get("exam_id", "").strip()
        roll_number  = request.args.get("roll_number", "").strip()

        if not student_name or not exam_id:
            return jsonify({"error": "student_name and exam_id are required"}), 400

        buffer, exam_name = _build_report_pdf(student_name, int(exam_id), roll_number)
        if buffer is None:
            return jsonify({"error": "No submissions found for this student in this exam"}), 404

        safe_name  = re.sub(r"[^\w\s-]", "", student_name).replace(" ", "_")
        safe_exam  = re.sub(r"[^\w\s-]", "", exam_name).replace(" ", "_")
        filename   = f"{safe_name}_{safe_exam}_Report.pdf"

        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf"
        )

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500


# ── ROUTE: Teacher lists all students who submitted for an exam ──
@app.route("/list_submissions", methods=["GET"])
def list_submissions():
    """
    GET /list_submissions?exam_id=1
    Returns list of unique students who submitted for an exam,
    with their total marks. Used by teacher to see all downloadable reports.
    """
    try:
        exam_id = request.args.get("exam_id", "").strip()
        if not exam_id:
            return jsonify({"error": "exam_id is required"}), 400

        conn   = sqlite3.connect("grading_system.db")
        cursor = conn.cursor()

        cursor.execute("SELECT exam_name FROM exams WHERE exam_id = ?", (exam_id,))
        row = cursor.fetchone()
        exam_name = row[0] if row else f"Exam {exam_id}"

        cursor.execute("""
            SELECT s.student_name,
                   ROUND(SUM(s.awarded_marks), 2),
                   ROUND(SUM(q.max_marks), 2),
                   u.roll_number
            FROM submissions s
            JOIN questions q ON s.question_id = q.question_id
            LEFT JOIN users u ON u.full_name = s.student_name
            WHERE s.exam_id = ?
            GROUP BY s.student_name
            ORDER BY SUM(s.awarded_marks) DESC
        """, (exam_id,))

        students = [
            {
                "student_name" : r[0],
                "total_awarded": r[1],
                "total_max"    : r[2],
                "percentage"   : round((r[1]/r[2])*100, 1) if r[2] else 0,
                "roll_number"  : r[3] or "—"
            }
            for r in cursor.fetchall()
        ]

        # ── AI Feedback Status (teacher-only signal) ───────
        # For each student, tell the teacher how many essay answers
        # actually got real Groq-generated feedback ("ai") versus how
        # many silently fell back to the generic template ("fallback",
        # e.g. because the Groq API call failed or timed out). Students
        # never see this — it's surfaced only on the teacher Reports tab.
        cursor.execute("""
            SELECT s.student_name, s.feedback
            FROM submissions s
            JOIN questions q ON s.question_id = q.question_id
            WHERE s.exam_id = ? AND q.question_type = 'essay'
        """, (exam_id,))

        ai_status = {}  # student_name -> {"ai": n, "fallback": n, "total": n}
        for student_name, feedback_json in cursor.fetchall():
            entry = ai_status.setdefault(student_name, {"ai": 0, "fallback": 0, "total": 0})
            entry["total"] += 1
            try:
                fb = json.loads(feedback_json) if feedback_json else {}
                if fb.get("feedback_source") == "ai":
                    entry["ai"] += 1
                else:
                    entry["fallback"] += 1
            except Exception:
                entry["fallback"] += 1

        for s in students:
            counts = ai_status.get(s["student_name"], {"ai": 0, "fallback": 0, "total": 0})
            s["essay_count"]        = counts["total"]
            s["ai_feedback_count"]  = counts["ai"]
            s["fallback_count"]     = counts["fallback"]

        conn.close()

        return jsonify({"status": "success", "exam_name": exam_name, "students": students})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)