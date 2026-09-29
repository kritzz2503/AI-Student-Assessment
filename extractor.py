# extractor.py
import os
import re
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"

def extract_text_from_file(filepath):
    """
    Detects file type and extracts raw text.
    Supports: .txt, .pdf, .docx, .png, .jpg
    """
    ext = os.path.splitext(filepath)[1].lower()

    # ── Plain Text ─────────────────────────────────────
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    # ── Word Document ──────────────────────────────────
    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            return f"Error reading docx: {str(e)}"

    # ── PDF ────────────────────────────────────────────
    elif ext == ".pdf":
        try:
            import PyPDF2
            text = ""
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # If no text found → scanned PDF → use OCR
            if not text.strip():
                try:
                    from pdf2image import convert_from_path
                    images = convert_from_path(filepath, poppler_path=POPPLER_PATH)
                    for img in images:
                        text += pytesseract.image_to_string(img) + "\n"
                except Exception as e:
                    text = f"OCR error: {str(e)}"

            return text

        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    # ── Image (handwritten/scanned) ────────────────────
    elif ext in [".png", ".jpg", ".jpeg"]:
        try:
            img = Image.open(filepath)
            return pytesseract.image_to_string(img)
        except Exception as e:
            return f"OCR error: {str(e)}"

    else:
        return f"Unsupported file type: {ext}"


def parse_response_sheet(raw_text):
    """
    Splits extracted text into per-question answers.
    Student writes answers like:
        Q1: A
        Q2: Plants make food using sunlight
        Q3: Technology has changed education...
    """
    # Try pattern: Q1: answer
    pattern = r"[Qq](\d+)[:.)\s]+(.+?)(?=[Qq]\d+[:.)\s]|\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL)

    answers = {}
    for q_num, answer_text in matches:
        answers[int(q_num)] = answer_text.strip()

    # If nothing matched, try splitting by lines as fallback
    if not answers:
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        for i, line in enumerate(lines, 1):
            answers[i] = line

    return answers

#import pytesseract

#pytesseract.pytesseract.tesseract_cmd = (
    #r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#)

#POPPLER_PATH = r"C:\Users\msubr\Downloads\Release-26.02.0-0.zip\poppler-26.02.0\Library\bin"