"""
UChicago Academic Advising Chatbot

Loads scraped program/course data and uses OpenAI to answer
academic advising questions with relevant context injection.
"""

import json
import os

from openai import OpenAI

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SYSTEM_PROMPT = """\
You are an experienced academic advisor for the University of Chicago's undergraduate College. \
Your role is to help students understand degree requirements, plan their course schedules, \
and make well-informed academic decisions.

## UChicago Academic Context
- UChicago uses the **quarter system**: Autumn, Winter, Spring (and optional Summer).
- Students must complete the **Core Curriculum** (general education requirements) alongside their major.
- Course codes follow the format DEPT NNNNN (e.g., CMSC 15400, MATH 20300).
- Students may declare one or more majors and/or minors.
- Most majors require a set of required courses plus electives from an approved list.

## Response Formatting Instructions
Format your responses using Markdown for readability:
- Use **bold** for course codes and important terms (e.g., **CMSC 15400**)
- Use bullet lists for listing courses or requirements
- Use ### headers to organize sections when answering detailed questions
- Keep responses focused and well-structured
- Use numbered lists for sequential recommendations (e.g., suggested course order)

## Advising Approach
- When a student asks about requirements, list them clearly with course codes and titles.
- When recommending courses, consider prerequisites and typical sequencing.
- For "what should I take next" questions, consider what the student has already completed.
- For double major feasibility questions, identify overlapping requirements.
- If a student has uploaded their transcript, reference their completed courses and \
  identify remaining requirements.

## Guardrails
- Only answer questions related to academics, courses, and university planning.
- If asked about non-academic topics, politely redirect: \
  "I'm here to help with academic advising at UChicago! Feel free to ask me about \
  courses, majors, minors, or planning your schedule."
- Always recommend that students verify critical decisions with their official \
  departmental advisor or the College Advising office.
- If the data doesn't cover something, be honest and suggest checking the official catalog.\
"""

NO_DATA_NOTE = (
    "\n\n[Note: No scraped catalog data is currently loaded. "
    "Please run `python scraper.py --test` or `python scraper.py --all` first. "
    "I'll do my best with general UChicago knowledge, but my answers may not "
    "reflect the latest catalog requirements.]"
)


class Chatbot:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        self.model = model
        self.programs = {}
        self.courses = {}
        self._program_keywords = {}  # slug -> set of lowercase keywords
        self._load_data()

    def _load_data(self):
        """Load scraped JSON data from the data/ directory."""
        programs_path = os.path.join(DATA_DIR, "programs.json")
        courses_path = os.path.join(DATA_DIR, "courses.json")

        if os.path.exists(programs_path):
            with open(programs_path) as f:
                self.programs = json.load(f)

        if os.path.exists(courses_path):
            with open(courses_path) as f:
                self.courses = json.load(f)

        # Build keyword index for context matching
        for slug, prog in self.programs.items():
            keywords = set()
            name = prog.get("name", "").lower()
            keywords.update(name.split())
            # Add the slug itself as a keyword
            keywords.add(slug.lower())
            # Add common abbreviations
            abbrevs = {
                "computerscience": {"cs", "computer", "compsci", "cmsc"},
                "economics": {"econ"},
                "mathematics": {"math", "maths"},
                "physics": {"phys"},
                "statistics": {"stats", "stat"},
                "biologicalsciences": {"bio", "biology"},
                "chemistry": {"chem"},
                "english": {"english", "engl"},
                "history": {"hist"},
                "philosophy": {"phil"},
                "politicalscience": {"polisci", "poli", "political"},
                "psychology": {"psych"},
                "sociology": {"soc", "socio"},
            }
            if slug in abbrevs:
                keywords.update(abbrevs[slug])
            # Remove stop words
            keywords.discard("and")
            keywords.discard("of")
            keywords.discard("the")
            keywords.discard("in")
            self._program_keywords[slug] = keywords

    def _find_relevant_programs(self, message):
        """Find programs relevant to the user's message via keyword matching."""
        msg_lower = message.lower()
        msg_words = set(msg_lower.split())

        scored = []
        for slug, keywords in self._program_keywords.items():
            # Score: count how many keywords appear in the message
            score = 0
            for kw in keywords:
                if kw in msg_words or kw in msg_lower:
                    score += 1
            if score > 0:
                scored.append((slug, score))

        # Sort by score descending, return top matches
        scored.sort(key=lambda x: x[1], reverse=True)
        return [slug for slug, score in scored[:3]]

    def _build_context(self, message, completed_courses=None):
        """Build context string with relevant program/course data."""
        relevant_slugs = self._find_relevant_programs(message)
        completed_set = set(completed_courses) if completed_courses else set()

        if not relevant_slugs and self.programs:
            # If no specific program matched, provide a list of available programs
            program_names = [p["name"] for p in self.programs.values()]
            return (
                "\n\nAvailable programs in the catalog:\n"
                + "\n".join(f"- {name}" for name in sorted(program_names))
                + "\n\n(Ask about a specific program for detailed requirements.)"
            )

        context_parts = []
        mentioned_codes = set()

        for slug in relevant_slugs:
            prog = self.programs[slug]
            part = f"\n\n## {prog['name']}\n"
            if prog.get("description"):
                part += f"{prog['description'][:500]}\n"

            if prog.get("requirements"):
                part += "\n### Requirements:\n"
                for section in prog["requirements"]:
                    if section.get("header"):
                        part += f"\n**{section['header']}**\n"
                    for course in section.get("courses", []):
                        prefix = ""
                        if course.get("or_alternative"):
                            prefix = "  OR "
                        elif course.get("elective_option"):
                            prefix = "  - "
                        units = f" ({course['units']})" if course.get("units") else ""
                        code = course.get("code", "")
                        completed_marker = " [COMPLETED]" if code in completed_set else ""
                        part += f"{prefix}{code}: {course['title']}{units}{completed_marker}\n"
                        if code:
                            mentioned_codes.add(code)

            context_parts.append(part)

        # Add details for mentioned courses
        course_details = []
        for code in mentioned_codes:
            c = self.courses.get(code)
            if c and c.get("description"):
                detail = f"\n**{code} - {c.get('name', '')}** ({c.get('units', '?')} units)"
                if c.get("description"):
                    detail += f"\n{c['description'][:200]}"
                if c.get("details", {}).get("prerequisites"):
                    detail += f"\nPrerequisites: {c['details']['prerequisites']}"
                if c.get("details", {}).get("terms_offered"):
                    detail += f"\nTerms: {c['details']['terms_offered']}"
                course_details.append(detail)

        if course_details:
            context_parts.append("\n### Course Details:\n" + "\n".join(course_details[:20]))

        return "".join(context_parts)

    def chat(self, message, conversation_history=None, completed_courses=None):
        """Send a message and get a response, with context injection.

        Args:
            message: The user's message string.
            conversation_history: List of prior {"role": ..., "content": ...} dicts.
            completed_courses: List of course code strings from uploaded transcript.

        Returns:
            The assistant's response string.
        """
        if conversation_history is None:
            conversation_history = []
        if completed_courses is None:
            completed_courses = []

        # Build system prompt with injected context
        system_content = SYSTEM_PROMPT
        if not self.programs:
            system_content += NO_DATA_NOTE
        else:
            context = self._build_context(message, completed_courses)
            if context:
                system_content += "\n\n--- CATALOG DATA ---" + context

        # Inject completed courses section if available
        if completed_courses:
            courses_list = ", ".join(sorted(completed_courses))
            system_content += (
                f"\n\n--- COMPLETED COURSES (from student transcript) ---\n"
                f"The student has completed the following courses: {courses_list}\n"
                f"When discussing requirements, note which ones are already fulfilled. "
                f"When recommending next courses, skip prerequisites they've already completed "
                f"and suggest the logical next steps."
            )

        messages = [{"role": "system", "content": system_content}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )

        return response.choices[0].message.content
