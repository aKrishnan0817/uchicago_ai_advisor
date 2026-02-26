"""
UChicago Academic Advising Chatbot

Loads scraped program/course data and uses OpenAI to answer
academic advising questions with relevant context injection.
"""

import json
import os
import re

from openai import OpenAI

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SYSTEM_PROMPT = """\
You are an experienced academic advisor for the University of Chicago's undergraduate College. \
Your role is to help students understand degree requirements, plan their course schedules, \
and make well-informed academic decisions.

## UChicago Academic Context
- UChicago uses the **quarter system**: Autumn, Winter, Spring (and optional Summer).
- Students must complete the **Core Curriculum** alongside their major(s).
- Course codes follow the format DEPT NNNNN (e.g., CMSC 15400, MATH 20300).
- Students may declare one or more majors and/or minors.

## Core Curriculum Areas
Students must complete requirements in ALL of these areas:
1. **Humanities** (HUMA sequence — 6 quarters)
2. **Civilization Studies** (2 quarters — e.g., CRES, TURK, SOSC-based civ sequences)
3. **Arts** (2 courses — e.g., ARTV, MUSI, TAPS, CRWR courses)
4. **Social Sciences** (3 quarters of a Social Sciences sequence)
5. **Natural Sciences** (2 quarters of a physical science + 2 quarters of a biological science)
6. **Mathematical Sciences** (MATH 13100-13200-13300 or MATH 15100-15200-15300 or MATH 16100-16200-16300)
7. **Language Competency** (demonstrated via placement or coursework)
8. **Physical Education** (6 quarters)

When a student has transcript data, note which Core areas appear to be satisfied and which remain.

## STRICT DATA GROUNDING RULES
- ONLY reference course codes that appear in the CATALOG DATA below. NEVER invent or guess course codes.
- If a course code is not in the provided data, explicitly say "I don't have that course in my catalog data."
- When stating a course title, use ONLY the title from the catalog data. Do not paraphrase or guess titles.

## Prerequisite Checking
- Before recommending a course, check its prerequisites in the catalog data.
- **COMPLETED** courses (grade received) DO satisfy prerequisites.
- **IN-PROGRESS** courses (currently enrolled, no grade yet) do NOT satisfy prerequisites.
- If a recommended course has prerequisites the student hasn't completed, warn them clearly.

## Course Status Rules
- **[COMPLETED]** = grade received. Satisfies prerequisites. Counts toward requirements.
- **[IN PROGRESS]** = currently enrolled, no grade yet. Does NOT satisfy prerequisites.
- **[NOT STARTED]** = not yet taken.
- Always distinguish these statuses clearly when discussing a student's progress.

## Multi-Major Overlap & Option-Maximizing Advice
- When a student discusses multiple majors/minors, ALWAYS identify shared courses that count toward more than one program.
- Present a dedicated **Cross-Program Overlap** section showing which courses satisfy multiple programs.
- For undecided students, prioritize recommending courses that keep the most options open (e.g., "MATH 20250 counts toward Mathematics, CAAM, and Statistics, keeping all three paths open").

## Anti-Filler Rule
- Do NOT use generic filler like "consult your advisor", "review course descriptions", or "consider your interests."
- Give concrete, specific recommendations based on the data.
- You may add ONE brief disclaimer at the very end if critical (e.g., "Confirm enrollment dates with the registrar.").

## Response Formatting
Format your responses using Markdown for readability:
- Use **bold** for course codes and important terms (e.g., **CMSC 15400**)
- Use bullet lists for listing courses or requirements
- Use ### headers to organize sections when answering detailed questions
- Keep responses focused and well-structured
- Use numbered lists for sequential recommendations (e.g., suggested course order)

## Guardrails
- Only answer questions related to academics, courses, and university planning.
- If asked about non-academic topics, politely redirect: \
  "I'm here to help with academic advising at UChicago! Feel free to ask me about \
  courses, majors, minors, or planning your schedule."
- If the data doesn't cover something, be honest and suggest checking the official catalog.\
"""

NO_DATA_NOTE = (
    "\n\n[Note: No scraped catalog data is currently loaded. "
    "Please run `python scraper.py --test` or `python scraper.py --all` first. "
    "I'll do my best with general UChicago knowledge, but my answers may not "
    "reflect the latest catalog requirements.]"
)

# Mapping from informal department abbreviations to official codes
DEPT_ABBREVS = {
    "cs": "CMSC",
    "cmsc": "CMSC",
    "compsci": "CMSC",
    "math": "MATH",
    "stat": "STAT",
    "stats": "STAT",
    "econ": "ECON",
    "phys": "PHYS",
    "chem": "CHEM",
    "bio": "BIOS",
    "bios": "BIOS",
    "phil": "PHIL",
    "psych": "PSYC",
    "polisci": "PLSC",
    "poli": "PLSC",
    "soc": "SOCI",
    "hist": "HIST",
    "engl": "ENGL",
    "english": "ENGL",
    "art": "ARTV",
    "artv": "ARTV",
    "musi": "MUSI",
    "music": "MUSI",
    "ling": "LING",
    "anth": "ANTH",
    "sosc": "SOSC",
    "huma": "HUMA",
    "taps": "TAPS",
    "caam": "CAAM",
    "turk": "NEHC",
}


def _normalize_code(code):
    """Normalize course codes by replacing non-breaking spaces with regular spaces."""
    return code.replace('\xa0', ' ').strip()


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
                "caam": {"caam", "computational", "applied"},
                "visualarts": {"art", "arts", "artv", "visual"},
            }
            if slug in abbrevs:
                keywords.update(abbrevs[slug])
            # Remove stop words
            for stop in ("and", "of", "the", "in"):
                keywords.discard(stop)
            self._program_keywords[slug] = keywords

    def _find_relevant_programs(self, message, conversation_history=None):
        """Find programs relevant to the user's message via keyword matching.

        Also scans last 4 user messages from conversation history for keyword persistence.
        """
        msg_lower = message.lower()
        msg_words = set(msg_lower.split())

        # Also scan recent conversation history for persistent program references
        history_words = set()
        if conversation_history:
            user_msgs = [m["content"] for m in conversation_history if m.get("role") == "user"]
            for msg in user_msgs[-4:]:
                history_words.update(msg.lower().split())

        combined_words = msg_words | history_words

        scored = []
        for slug, keywords in self._program_keywords.items():
            score = 0
            for kw in keywords:
                # Current message matches weighted higher
                if kw in msg_words or kw in msg_lower:
                    score += 2
                # History matches weighted lower
                elif kw in history_words:
                    score += 1
            if score > 0:
                scored.append((slug, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [slug for slug, score in scored[:8]]

    def _resolve_course_reference(self, message):
        """Resolve informal course code references to official catalog codes.

        Handles patterns like 'cs143', 'math203', 'stat244' by padding to 5 digits
        and looking up in the course catalog.

        Returns dict of {"cs143": "CMSC 14300", ...} for matches found.
        """
        resolved = {}
        # Match patterns like cs143, math203, stat24300, CMSC154
        pattern = r'\b([a-zA-Z]{2,6})[\s-]?(\d{2,5})\b'
        for match in re.finditer(pattern, message, re.IGNORECASE):
            dept_raw = match.group(1).lower()
            num_raw = match.group(2)
            original = match.group(0)

            # Skip if it's already a proper 5-digit code with known dept
            if len(num_raw) == 5 and dept_raw.upper() in {c.split()[0] for c in self.courses}:
                continue

            # Map informal dept to official code
            official_dept = DEPT_ABBREVS.get(dept_raw, dept_raw.upper())

            # Pad number to 5 digits
            if len(num_raw) <= 3:
                padded = num_raw + "0" * (5 - len(num_raw))
            else:
                padded = num_raw.ljust(5, "0")

            candidate = f"{official_dept} {padded}"
            if candidate in self.courses:
                resolved[original] = candidate

        return resolved

    def _search_courses_by_name(self, query, limit=5):
        """Search courses by substring match against course names.

        Returns list of (code, name) tuples for matches.
        """
        query_lower = query.lower()
        results = []
        for code, course in self.courses.items():
            name = course.get("name", "").lower()
            if query_lower in name:
                results.append((code, course.get("name", "")))
                if len(results) >= limit:
                    break
        return results

    def _check_prerequisites(self, code, completed_set, in_progress_set):
        """Check if prerequisites for a course are satisfied.

        Returns (satisfied: bool, explanation: str).
        """
        course = self.courses.get(code)
        if not course:
            return True, ""

        details = course.get("details", {})
        prereq_text = details.get("prerequisites", "")

        # Fallback: some courses have prereqs concatenated into terms_offered or instructors
        if not prereq_text:
            for field in ("terms_offered", "instructors"):
                field_val = details.get(field, "")
                if "Prerequisite" in field_val:
                    idx = field_val.index("Prerequisite")
                    prereq_text = field_val[idx:]
                    break

        if not prereq_text:
            return True, ""

        # Extract course codes mentioned in prereq text
        prereq_codes = set()
        for match in re.finditer(r'\b([A-Z]{2,5})\s+(\d{5})\b', prereq_text):
            prereq_codes.add(f"{match.group(1)} {match.group(2)}")

        if not prereq_codes:
            return True, f"Prerequisites: {prereq_text}"

        missing = []
        in_prog = []
        for pc in prereq_codes:
            if pc in completed_set:
                continue
            elif pc in in_progress_set:
                in_prog.append(pc)
            else:
                missing.append(pc)

        if not missing and not in_prog:
            return True, ""

        parts = []
        if missing:
            parts.append(f"Not completed: {', '.join(sorted(missing))}")
        if in_prog:
            parts.append(f"In-progress (not yet satisfied): {', '.join(sorted(in_prog))}")

        return False, f"Prereq issue for {code}: {'; '.join(parts)}"

    def _build_context(self, message, completed_courses=None, in_progress_courses=None,
                       conversation_history=None):
        """Build context string with relevant program/course data."""
        relevant_slugs = self._find_relevant_programs(message, conversation_history)
        completed_set = set(completed_courses) if completed_courses else set()
        in_progress_set = set(in_progress_courses) if in_progress_courses else set()

        if not relevant_slugs and self.programs:
            program_names = [p["name"] for p in self.programs.values()]
            return (
                "\n\nAvailable programs in the catalog:\n"
                + "\n".join(f"- {name}" for name in sorted(program_names))
                + "\n\n(Ask about a specific program for detailed requirements.)"
            )

        context_parts = []
        mentioned_codes = set()
        # For cross-program overlap: {normalized_code: set(program_names)}
        code_to_programs = {}

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
                        raw_code = course.get("code", "")
                        code = _normalize_code(raw_code)

                        if code in completed_set:
                            status_marker = " [COMPLETED]"
                        elif code in in_progress_set:
                            status_marker = " [IN PROGRESS]"
                        else:
                            status_marker = ""

                        part += f"{prefix}{code}: {course['title']}{units}{status_marker}\n"
                        if code:
                            mentioned_codes.add(code)
                            # Track for overlap analysis
                            if code not in code_to_programs:
                                code_to_programs[code] = set()
                            code_to_programs[code].add(prog['name'])

            context_parts.append(part)

        # Cross-program overlap section
        if len(relevant_slugs) >= 2:
            overlaps = {code: progs for code, progs in code_to_programs.items()
                        if len(progs) >= 2}
            if overlaps:
                overlap_section = "\n\n### Cross-Program Overlap\n"
                overlap_section += "These courses count toward multiple programs:\n"
                for code in sorted(overlaps.keys()):
                    progs = sorted(overlaps[code])
                    course_obj = self.courses.get(code)
                    name = course_obj.get("name", "") if course_obj else ""
                    status = ""
                    if code in completed_set:
                        status = " [COMPLETED]"
                    elif code in in_progress_set:
                        status = " [IN PROGRESS]"
                    overlap_section += f"- **{code}**: {name} → counts for: {', '.join(progs)}{status}\n"
                context_parts.append(overlap_section)

        # Add details for mentioned courses
        course_details = []
        prereq_warnings = []
        for code in mentioned_codes:
            c = self.courses.get(code)
            if c and c.get("description"):
                detail = f"\n**{code} - {c.get('name', '')}** ({c.get('units', '?')} units)"
                if c.get("description"):
                    detail += f"\n{c['description'][:200]}"
                if c.get("details", {}).get("prerequisites"):
                    detail += f"\nPrerequisites: {c['details']['prerequisites']}"
                elif c.get("details", {}).get("terms_offered"):
                    terms = c["details"]["terms_offered"]
                    if "Prerequisite" in terms:
                        detail += f"\n{terms}"
                if c.get("details", {}).get("terms_offered"):
                    terms = c["details"]["terms_offered"]
                    if "Prerequisite" not in terms:
                        detail += f"\nTerms: {terms}"
                course_details.append(detail)

            # Prereq check for courses not yet completed
            if code not in completed_set and code not in in_progress_set:
                satisfied, explanation = self._check_prerequisites(
                    code, completed_set, in_progress_set)
                if not satisfied:
                    prereq_warnings.append(explanation)

        if course_details:
            context_parts.append("\n### Course Details:\n" + "\n".join(course_details[:25]))

        if prereq_warnings:
            context_parts.append(
                "\n\n### Prerequisite Warnings:\n" +
                "\n".join(f"- {w}" for w in prereq_warnings)
            )

        # Resolve informal course references (e.g., "cs143" → "CMSC 14300")
        resolved = self._resolve_course_reference(message)
        if resolved:
            resolution_section = "\n\n### Resolved Course References:\n"
            for informal, official in resolved.items():
                c = self.courses.get(official)
                name = c.get("name", "") if c else ""
                resolution_section += f'- "{informal}" → **{official}**: {name}\n'
                # Add details for resolved courses if not already mentioned
                if official not in mentioned_codes and c:
                    detail = f"\n**{official} - {name}** ({c.get('units', '?')} units)"
                    if c.get("description"):
                        detail += f"\n{c['description'][:200]}"
                    if c.get("details", {}).get("prerequisites"):
                        detail += f"\nPrerequisites: {c['details']['prerequisites']}"
                    elif c.get("details", {}):
                        for field in ("terms_offered", "instructors"):
                            fv = c["details"].get(field, "")
                            if "Prerequisite" in fv:
                                idx = fv.index("Prerequisite")
                                detail += f"\n{fv[idx:]}"
                                break
                    resolution_section += detail + "\n"
            context_parts.append(resolution_section)

        # Search for courses by name if message contains quoted terms or
        # natural language references
        name_queries = re.findall(r'"([^"]+)"', message)
        # Also try phrases after "like", "called", "named", "about"
        for match in re.finditer(
                r'\b(?:like|called|named|about|such as)\s+["\']?([a-zA-Z][a-zA-Z\s]{2,30})["\']?',
                message, re.IGNORECASE):
            candidate = match.group(1).strip().rstrip('.,?!')
            if candidate and len(candidate) > 2:
                name_queries.append(candidate)

        if name_queries:
            search_results = []
            for q in name_queries[:3]:
                results = self._search_courses_by_name(q, limit=5)
                for code, name in results:
                    if code not in mentioned_codes:
                        search_results.append(f"- **{code}**: {name}")
                        mentioned_codes.add(code)
            if search_results:
                context_parts.append(
                    "\n\n### Course Name Search Results:\n" +
                    "\n".join(search_results[:10])
                )

        return "".join(context_parts)

    def chat(self, message, conversation_history=None, completed_courses=None,
             in_progress_courses=None):
        """Send a message and get a response, with context injection.

        Args:
            message: The user's message string.
            conversation_history: List of prior {"role": ..., "content": ...} dicts.
            completed_courses: List of course code strings the student has completed.
            in_progress_courses: List of course code strings currently in progress.

        Returns:
            The assistant's response string.
        """
        if conversation_history is None:
            conversation_history = []
        if completed_courses is None:
            completed_courses = []
        if in_progress_courses is None:
            in_progress_courses = []

        # Build system prompt with injected context
        system_content = SYSTEM_PROMPT
        if not self.programs:
            system_content += NO_DATA_NOTE
        else:
            context = self._build_context(
                message,
                completed_courses=completed_courses,
                in_progress_courses=in_progress_courses,
                conversation_history=conversation_history,
            )
            if context:
                system_content += "\n\n--- CATALOG DATA ---" + context

        # Inject transcript course sections
        if completed_courses or in_progress_courses:
            system_content += "\n\n--- STUDENT TRANSCRIPT ---\n"
            if completed_courses:
                courses_list = ", ".join(sorted(completed_courses))
                system_content += (
                    f"**COMPLETED** (grade received — these satisfy prerequisites):\n"
                    f"{courses_list}\n\n"
                )
            if in_progress_courses:
                ip_list = ", ".join(sorted(in_progress_courses))
                system_content += (
                    f"**IN PROGRESS** (currently enrolled — these do NOT satisfy prerequisites):\n"
                    f"{ip_list}\n\n"
                )
            system_content += (
                "When discussing requirements, clearly mark which are fulfilled, "
                "in-progress, and remaining. When recommending next courses, "
                "remember that in-progress courses have NOT yet satisfied prerequisites."
            )

        messages = [{"role": "system", "content": system_content}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=2500,
        )

        return response.choices[0].message.content
