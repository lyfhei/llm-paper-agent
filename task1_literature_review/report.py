from datetime import date
from pathlib import Path
from typing import List, Dict

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import REPORTS_DIR


def generate_report(papers: List[Dict]) -> str:
    today = date.today().strftime("%Y-%m-%d")
    lines = [
        f"# LLM Paper Report — {today}",
        "",
        f"**Total evaluated:** {len(papers)} papers  |  **Period:** Last 30 days  |  **Source:** arXiv",
        "",
        "## Executive Summary",
        "",
        f"Ranked {len(papers)} recent LLM-related arXiv papers by novelty and practicality (scored by Claude). "
        "Citation counts are shown as auxiliary information only (recent papers have few citations). "
        "Each paper is annotated with a suggested experiment patch for Task 2.",
        "",
        "---",
        "",
        "## Top Papers",
        "",
    ]

    for rank, p in enumerate(papers, 1):
        confidence = p.get("confidence", 0.0)

        lines += [
            f"### #{rank} — {p['title']}",
            "",
            f"**Authors:** {', '.join(p['authors'])}  ",
            f"**Submitted:** {p['submitted_date']}  ",
            f"**Score:** {p['composite_score']} (novelty={p['novelty']}, practicality={p['practicality']}, clarity={p['clarity']})",
            "",
            f"> {p['summary']}",
            "",
            f"**Target:** {p.get('target_component', 'other')}  |  **Applicability:** {p.get('applicability', 'low')}  |  **Confidence:** {confidence:.2f}  ",
            f"**Link:** {p['url']}",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


def save_report(content: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")
    path = REPORTS_DIR / f"{today}.md"
    path.write_text(content, encoding="utf-8")
    print(f"[report] Saved to {path}")
    return path
