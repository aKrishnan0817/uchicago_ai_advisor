"""
UChicago College Catalog Scraper

Scrapes program requirements and course details from the UChicago
College Catalog website. Outputs to data/programs.json and data/courses.json.

Usage:
    python scraper.py --test    # Scrape 5 test programs
    python scraper.py --all     # Scrape all programs
"""

import argparse
import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "http://collegecatalog.uchicago.edu"
PROGRAMS_INDEX = f"{BASE_URL}/thecollege/programsofstudy/"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

HEADERS = {
    "User-Agent": (
        "UChicago-AcademicAdvisor-Bot/1.0 "
        "(educational project; polite crawling with delays)"
    )
}

TEST_SLUGS = {
    "computerscience",
    "economics",
    "mathematics",
    "physics",
    "statistics",
}

DELAY_SECONDS = 1.5


def fetch_page(url):
    """Fetch a page and return a BeautifulSoup object."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


# ---------------------------------------------------------------------------
# Phase A: Discover programs
# ---------------------------------------------------------------------------

def discover_programs():
    """Scrape the programs-of-study index page to get all program names + URLs."""
    soup = fetch_page(PROGRAMS_INDEX)

    programs = []
    # The catalog sidebar navigation uses ul.nav.leveltwo for program links
    nav = soup.select("ul.nav.leveltwo li a")
    if not nav:
        # Fallback: try links inside the main content area
        nav = soup.select("#defined a[href*='/thecollege/']")
    if not nav:
        # Broader fallback: any link under the main content
        nav = soup.select("#textcontainer a[href*='/thecollege/']")

    for link in nav:
        href = link.get("href", "")
        name = link.get_text(strip=True)
        if not name or not href:
            continue
        # Skip links back to the index itself
        if href.rstrip("/").endswith("programsofstudy"):
            continue
        # Build absolute URL
        if href.startswith("/"):
            full_url = BASE_URL + href
        else:
            full_url = href
        # Extract slug for filtering
        slug = href.rstrip("/").split("/")[-1]
        programs.append({"name": name, "slug": slug, "url": full_url})

    # Deduplicate by slug
    seen = set()
    unique = []
    for p in programs:
        if p["slug"] not in seen:
            seen.add(p["slug"])
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# Phase B: Scrape a single program page
# ---------------------------------------------------------------------------

def parse_requirement_tables(soup):
    """Parse sc_courselist tables into structured requirement sections."""
    sections = []
    tables = soup.select("table.sc_courselist")

    for table in tables:
        current_section = {"header": None, "courses": []}
        rows = table.select("tr")

        for row in rows:
            # Check for section header
            comment = row.select_one("span.courselistcomment")
            if comment:
                # Save previous section if it has courses
                if current_section["courses"]:
                    sections.append(current_section)
                current_section = {
                    "header": comment.get_text(strip=True),
                    "courses": [],
                }
                continue

            # Check for area header row (areaheader class)
            areaheader = row.select_one("td.areaheader")
            if areaheader:
                if current_section["courses"]:
                    sections.append(current_section)
                current_section = {
                    "header": areaheader.get_text(strip=True),
                    "courses": [],
                }
                continue

            # Extract course code + title
            code_el = row.select_one("td.codecol a.bubblelink")
            if not code_el:
                code_el = row.select_one("td.codecol")
            if not code_el:
                continue

            code = code_el.get_text(strip=True)
            # Some codes span two columns (cross-listed)
            code2_el = row.select_one("td.codecol + td.codecol")
            if code2_el:
                code2 = code2_el.get_text(strip=True)
                if code2:
                    code = f"{code} / {code2}"

            title_el = row.select("td")
            title = ""
            for td in title_el:
                if "codecol" not in td.get("class", []) and "hourscol" not in td.get("class", []):
                    title = td.get_text(strip=True)
                    break

            units_el = row.select_one("td.hourscol")
            units = units_el.get_text(strip=True) if units_el else ""

            is_or = "orclass" in row.get("class", [])
            is_indented = bool(row.select_one("div[style*='margin-left']"))

            course_entry = {"code": code, "title": title, "units": units}
            if is_or:
                course_entry["or_alternative"] = True
            if is_indented:
                course_entry["elective_option"] = True

            current_section["courses"].append(course_entry)

        # Don't forget the last section
        if current_section["courses"]:
            sections.append(current_section)

    return sections


def parse_course_blocks(soup):
    """Parse courseblock divs for the course inventory on a program page."""
    courses = []
    blocks = soup.select("div.courseblock")

    for block in blocks:
        title_el = block.select_one("p.courseblocktitle strong")
        if not title_el:
            title_el = block.select_one("p.courseblocktitle")
        if not title_el:
            continue

        title_text = title_el.get_text(strip=True)

        # Parse: "CMSC 15100. Introduction to Computer Science I. 100 Units."
        match = re.match(
            r"^([A-Z]{2,5})\s+(\d{5})\.\s+(.+?)\.\s+(\d+)\s+Units?\.$",
            title_text,
        )
        if not match:
            # Try a looser pattern (some have variable units or different formatting)
            match = re.match(
                r"^([A-Z]{2,5})\s+(\d{5})\.\s+(.+?)\..*?(\d+)\s+Units?\.",
                title_text,
            )
        if not match:
            # Store even if we can't fully parse
            courses.append({
                "raw_title": title_text,
                "description": "",
                "details": {},
            })
            continue

        dept, number, name, units = match.groups()
        course_code = f"{dept} {number}"

        desc_el = block.select_one("p.courseblockdesc")
        description = desc_el.get_text(strip=True) if desc_el else ""

        # Parse detail fields
        details = {}
        detail_els = block.select("p.courseblockdetail")
        for det in detail_els:
            text = det.get_text(strip=True)
            # Common patterns: "Instructor(s): ...", "Terms Offered: ...",
            # "Prerequisite(s): ...", "Equivalent Course(s): ...", "Note(s): ..."
            for field in [
                "Instructor(s)",
                "Terms Offered",
                "Prerequisite(s)",
                "Equivalent Course(s)",
                "Note(s)",
            ]:
                if text.startswith(field):
                    val = text[len(field):].lstrip(":").strip()
                    key = (
                        field.replace("(s)", "s")
                        .replace("(", "")
                        .replace(")", "")
                        .lower()
                        .replace(" ", "_")
                    )
                    details[key] = val
                    break

        courses.append({
            "code": course_code,
            "department": dept,
            "number": number,
            "name": name,
            "units": int(units),
            "description": description,
            "details": details,
        })

    return courses


def scrape_program(name, url):
    """Scrape a single program page and return structured data."""
    print(f"  Scraping: {name} ({url})")
    soup = fetch_page(url)

    # Get the main page text content for context
    page_title_el = soup.select_one("#textcontainer h1, #content h1")
    page_title = page_title_el.get_text(strip=True) if page_title_el else name

    # Get introductory text (paragraphs before the first table or courseblock)
    intro_parts = []
    text_container = soup.select_one("#textcontainer")
    if text_container:
        for el in text_container.children:
            if hasattr(el, "name"):
                if el.name == "table" or (el.name == "div" and "courseblock" in el.get("class", [])):
                    break
                if el.name == "p":
                    text = el.get_text(strip=True)
                    if text:
                        intro_parts.append(text)

    requirements = parse_requirement_tables(soup)
    course_inventory = parse_course_blocks(soup)

    return {
        "name": page_title,
        "url": url,
        "description": " ".join(intro_parts[:5]),  # First few paragraphs
        "requirements": requirements,
        "course_count": len(course_inventory),
    }, course_inventory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape UChicago College Catalog")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="Scrape 5 test programs")
    group.add_argument("--all", action="store_true", help="Scrape all programs")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Phase A: Discovering programs...")
    programs = discover_programs()
    print(f"  Found {len(programs)} programs")

    if args.test:
        programs = [p for p in programs if p["slug"] in TEST_SLUGS]
        print(f"  Filtering to test set: {[p['name'] for p in programs]}")
        if not programs:
            print("  WARNING: No test programs matched. Using first 5 programs instead.")
            programs = discover_programs()[:5]

    print(f"\nPhase B: Scraping {len(programs)} program pages...")
    all_programs = {}
    all_courses = {}

    for i, prog in enumerate(programs):
        try:
            program_data, course_list = scrape_program(prog["name"], prog["url"])
            all_programs[prog["slug"]] = program_data

            for course in course_list:
                code = course.get("code")
                if code:
                    all_courses[code] = course

            if i < len(programs) - 1:
                time.sleep(DELAY_SECONDS)

        except Exception as e:
            print(f"  ERROR scraping {prog['name']}: {e}")
            continue

    # Save to JSON
    programs_path = os.path.join(DATA_DIR, "programs.json")
    courses_path = os.path.join(DATA_DIR, "courses.json")

    with open(programs_path, "w") as f:
        json.dump(all_programs, f, indent=2)
    print(f"\nSaved {len(all_programs)} programs to {programs_path}")

    with open(courses_path, "w") as f:
        json.dump(all_courses, f, indent=2)
    print(f"Saved {len(all_courses)} courses to {courses_path}")


if __name__ == "__main__":
    main()
