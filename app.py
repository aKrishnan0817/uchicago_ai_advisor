"""
Flask web app for the UChicago Academic Advising Chatbot.

Run with: python app.py
Then visit: http://localhost:5000
"""

import io
import re

import pdfplumber
from flask import Flask, render_template, request, jsonify

from chatbot import Chatbot

app = Flask(__name__)
bot = Chatbot()


def _parse_transcript(content):
    """Parse course codes and their statuses from transcript text.

    Attempts to distinguish completed courses (have a letter grade) from
    in-progress courses (marked 'In Progress' or no grade).

    Returns dict: {"completed": [...codes], "in_progress": [...codes]}
    """
    completed = set()
    in_progress = set()

    # Try structured parsing: lines with course code + grade info
    # Typical format: DEPT NNNNN ... (grade or 'In Progress or N/A')
    # Match course code at start, then look for grade at end of line
    structured_pattern = re.compile(
        r'\b([A-Z]{2,5})\s+(\d{5})\b'
    )

    lines = content.split('\n')
    has_grade_info = False

    for line in lines:
        code_match = structured_pattern.search(line)
        if not code_match:
            continue

        code = f"{code_match.group(1)} {code_match.group(2)}"

        # Check for grade indicators in the same line
        line_after_code = line[code_match.end():]

        if re.search(r'\bIn\s+Progress\b', line_after_code, re.IGNORECASE):
            in_progress.add(code)
            has_grade_info = True
        elif re.search(r'\bN/?A\b', line_after_code, re.IGNORECASE):
            in_progress.add(code)
            has_grade_info = True
        elif re.search(r'\b[A-D][+-]?\b', line_after_code) or re.search(r'\b[PS]\b', line_after_code):
            # Letter grade (A+, A, A-, B+, B, B-, C+, C, C-, D, P, S)
            completed.add(code)
            has_grade_info = True
        elif re.search(r'\bW\b', line_after_code):
            # Withdrawn — don't count
            has_grade_info = True
        else:
            # Code found but no grade info on this line
            in_progress.add(code)

    # If we never found any grade info, fall back to marking all as completed
    # (plain text file with just course codes)
    if not has_grade_info and (completed or in_progress):
        all_codes = completed | in_progress
        return {
            "completed": sorted(all_codes),
            "in_progress": [],
        }

    # If nothing was found at all via structured parsing, try simple extraction
    if not completed and not in_progress:
        matches = re.findall(r'\b([A-Z]{2,5})\s+(\d{5})\b', content)
        codes = sorted(set(f"{dept} {num}" for dept, num in matches))
        return {
            "completed": codes,
            "in_progress": [],
        }

    return {
        "completed": sorted(completed),
        "in_progress": sorted(in_progress),
    }


def _extract_pdf_text(file_storage):
    """Extract text from a PDF file upload using pdfplumber."""
    file_bytes = file_storage.read()
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status", methods=["GET"])
def status():
    """Return program/course counts and program names for the header info bar."""
    program_names = sorted(p["name"] for p in bot.programs.values())
    return jsonify({
        "program_count": len(bot.programs),
        "course_count": len(bot.courses),
        "program_names": program_names,
    })


@app.route("/upload-transcript", methods=["POST"])
def upload_transcript():
    """Receive a transcript file, parse course codes with status, and return them."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    # Only allow .txt, .csv, and .pdf files
    filename_lower = file.filename.lower()
    if not filename_lower.endswith((".txt", ".csv", ".pdf")):
        return jsonify({"error": "Only .txt, .csv, and .pdf files are supported"}), 400

    try:
        if filename_lower.endswith(".pdf"):
            content = _extract_pdf_text(file)
        else:
            content = file.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "Could not read file — ensure it is a text file"}), 400
    except Exception as e:
        return jsonify({"error": f"Could not read PDF: {e}"}), 400

    result = _parse_transcript(content)

    if not result["completed"] and not result["in_progress"]:
        return jsonify({"error": "No course codes found. Expected format: DEPT NNNNN (e.g., CMSC 14100)"}), 400

    return jsonify(result)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    # Conversation history from the frontend (list of {role, content} dicts)
    history = data.get("history", [])
    completed_courses = data.get("completed_courses", [])
    in_progress_courses = data.get("in_progress_courses", [])

    try:
        reply = bot.chat(
            message,
            conversation_history=history,
            completed_courses=completed_courses,
            in_progress_courses=in_progress_courses,
        )
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
