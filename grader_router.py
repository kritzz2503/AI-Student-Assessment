# grader_router.py
import json
from sklearn.metrics.pairwise import cosine_similarity

LLM_MODEL = "llama-3.3-70b-versatile"


def grade_mcq_answer(student_answer, correct_answer, max_marks):
    s = student_answer.strip().upper()[:1]
    is_correct = s == correct_answer.strip().upper()[:1]
    return (float(max_marks) if is_correct else 0.0), is_correct


def _llm_rubric_grade(groq_client, question_text, reference, student_answer, max_marks, kind):
    """Ask the LLM to grade against the question (and reference) using a strict rubric.
    Returns (fraction 0..1, reasoning) or None if the call fails."""
    if groq_client is None:
        return None
    ref_line = f"Reference / model answer: {reference}\n" if reference else ""
    if kind == "essay":
        rubric = ("Rubric (each 0-5): relevance to the question, content & ideas, "
                  "organisation & coherence, language & grammar. Off-topic or "
                  "nonsense answers must score very low. Do not reward length alone.")
    else:
        rubric = ("Rubric: award credit for each key idea from the reference answer that the "
                  "student conveys correctly, even if worded differently. Wrong or "
                  "irrelevant statements earn no credit.")
    prompt = f"""You are a strict but fair examiner.
Question: {question_text}
{ref_line}Student answer: \"\"\"{student_answer[:2500]}\"\"\"
{rubric}
Return ONLY JSON: {{"score_percent": <0-100 integer>, "reason": "<one short sentence>"}}"""
    try:
        r = groq_client.chat.completions.create(
            model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=150, temperature=0.0)
        raw = r.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        pct = max(0.0, min(100.0, float(data["score_percent"])))
        return pct / 100.0, str(data.get("reason", ""))[:300]
    except Exception as e:
        print(f"[grader] LLM grading failed, using model-only fallback: {e}")
        return None


def _similarity_fraction(similarity):
    if similarity >= 0.65:
        f = 0.85 + (similarity - 0.65) / 0.35 * 0.15
    elif similarity >= 0.45:
        f = 0.40 + (similarity - 0.45) / 0.20 * 0.45
    else:
        f = (similarity / 0.45) * 0.40 if similarity > 0 else 0.0
    return max(0.0, min(1.0, f))


def _level(pct):
    return ("Excellent" if pct >= 85 else "Good" if pct >= 70
            else "Partial" if pct >= 45 else "Needs Revision")


def grade_short_answer(student_answer, model_answer, max_marks, bert_model,
                       question_text="", groq_client=None):
    emb = bert_model.encode([student_answer, model_answer])
    similarity = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    sim_frac = _similarity_fraction(similarity)

    llm = _llm_rubric_grade(groq_client, question_text, model_answer, student_answer, max_marks, "short")
    if llm:
        # LLM judges meaning (70%), embedding similarity is a stabiliser (30%)
        fraction = 0.7 * llm[0] + 0.3 * sim_frac
        reason, method = llm[1], "hybrid (LLM rubric + semantic similarity)"
    else:
        fraction, reason, method = sim_frac, "", "semantic similarity only"
    awarded = round(fraction * float(max_marks) * 4) / 4  # nearest 0.25 mark
    return awarded, round(similarity, 4), _level(fraction * 100), reason, method


def grade_essay_answer(student_answer, predict_essay_score_fn, max_marks=6.0,
                       model_scale_max=6.0, question_text="", groq_client=None):
    raw = max(0.0, min(model_scale_max, float(predict_essay_score_fn(student_answer))))
    model_frac = raw / model_scale_max

    llm = _llm_rubric_grade(groq_client, question_text, "", student_answer, max_marks, "essay")
    if llm:
        # The ridge model never sees the question, so it can't judge relevance.
        # LLM (relevance + rubric) carries most weight; ridge acts as a style check.
        fraction = 0.75 * llm[0] + 0.25 * model_frac
        reason, method = llm[1], "hybrid (LLM rubric + trained essay model)"
    else:
        fraction, reason, method = model_frac, "", "trained essay model only"
    scaled = round(fraction * float(max_marks) * 4) / 4
    return scaled, raw, reason, method


def route_and_grade(question_type, student_answer, question_data,
                    bert_model=None, predict_essay_score_fn=None, groq_client=None):
    qtext = question_data.get("question_text", "")
    if question_type == "mcq":
        awarded, ok = grade_mcq_answer(student_answer, question_data["correct_answer"],
                                       question_data["max_marks"])
        return {"awarded_marks": awarded, "is_correct": ok}

    if question_type == "short_answer":
        awarded, sim, level, reason, method = grade_short_answer(
            student_answer, question_data["model_answer"], question_data["max_marks"],
            bert_model, qtext, groq_client)
        return {"awarded_marks": awarded, "similarity": sim, "level": level,
                "grading_reason": reason, "grading_method": method}

    if question_type == "essay":
        scaled, raw, reason, method = grade_essay_answer(
            student_answer, predict_essay_score_fn, question_data["max_marks"],
            question_text=qtext, groq_client=groq_client)
        return {"awarded_marks": scaled, "raw_score_out_of_6": raw,
                "grading_reason": reason, "grading_method": method}

    raise ValueError(f"Unknown question type: {question_type}")
