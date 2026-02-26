"""
Flask web app for the UChicago Academic Advising Chatbot.

Run with: python app.py
Then visit: http://localhost:5000
"""

import re

from flask import Flask, render_template, request, jsonify

from chatbot import Chatbot

app = Flask(__name__)
bot = Chatbot()


def _parse_transcript(content):
    """Parse course codes from transcript text.

    Matches patterns like 'CMSC 14100', 'MATH 15300', etc.
    Returns a sorted list of unique course code strings.
    """
    matches = re.findall(r'\b([A-Z]{2,5})\s*(\d{5})\b', content)
    codes = sorted(set(f"{dept} {num}" for dept, num in matches))
    return codes


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
    """Receive a transcript file, parse course codes, and return them."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    # Only allow .txt and .csv files
    if not file.filename.lower().endswith((".txt", ".csv")):
        return jsonify({"error": "Only .txt and .csv files are supported"}), 400

    try:
        content = file.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "Could not read file — ensure it is a text file"}), 400

    codes = _parse_transcript(content)
    if not codes:
        return jsonify({"error": "No course codes found. Expected format: DEPT NNNNN (e.g., CMSC 14100)"}), 400

    return jsonify({"courses": codes})


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

    try:
        reply = bot.chat(
            message,
            conversation_history=history,
            completed_courses=completed_courses,
        )
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
