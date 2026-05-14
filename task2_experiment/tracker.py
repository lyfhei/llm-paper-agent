import json
import time
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import EXPERIMENTS_DIR, REPORTS_DIR


def save_experiment(record: Dict) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    # Option C records have no patch_id — use "claude" as identifier
    patch_label = record.get("patch_id") or "claude"
    fname = f"{record['paper_id']}_{patch_label}.json"
    path = EXPERIMENTS_DIR / fname
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[tracker] Saved experiment record: {path}")
    return path


def _load_all() -> List[Dict]:
    return [
        json.loads(f.read_text(encoding="utf-8"))
        for f in sorted(EXPERIMENTS_DIR.glob("*.json"))
    ]


def summary() -> None:
    records = _load_all()
    if not records:
        print("[tracker] No experiment records found.")
        return

    paper_inspired = [r for r in records if r.get("experiment_type") == "paper-inspired"]
    config_ablation = [r for r in records if r.get("experiment_type") != "paper-inspired"]

    def _print_table(recs, title):
        print(f"\n{'='*70}")
        print(f"  {title} ({len(recs)} experiments)")
        print(f"{'='*70}")
        print(f"{'Paper/Change':<40} {'Rel.Impr':>9} {'Result'}")
        print("-" * 60)
        for r in sorted(recs, key=lambda x: x.get("relative_improvement", 0), reverse=True):
            label = r.get("change_description") or r.get("patch_id") or r.get("paper_id", "?")
            label = label[:39]
            ri = r.get("relative_improvement", 0)
            result = r.get("result", "?")
            print(f"{label:<40} {ri:>+9.3f} {result}")

    _print_table(paper_inspired, "Paper-Inspired (Option C — Claude-generated patches)")
    _print_table(config_ablation, "Config-Level Ablations")
    print()


def save_report(records: List[Dict] = None) -> Path:
    """Generate a Markdown report of all task2 experiments."""
    if records is None:
        records = _load_all()
    if not records:
        return None

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = time.strftime("%Y-%m-%d")
    path = REPORTS_DIR / f"task2_{date_str}.md"

    passes = [r for r in records if r.get("result") == "PASS"]
    fails  = [r for r in records if r.get("result") != "PASS"]

    lines = [
        f"# Task 2 — Agent Experiment Report ({date_str})",
        "",
        f"**Total experiments**: {len(records)} | "
        f"**PASS**: {len(passes)} | **FAIL**: {len(fails)}",
        "",
        "---",
        "",
    ]

    for r in sorted(records, key=lambda x: x.get("relative_improvement", 0), reverse=True):
        title = r.get("title") or r.get("paper_id", "unknown")
        ri = r.get("relative_improvement", 0)
        result = r.get("result", "?")
        symbol = "✅" if result == "PASS" else "❌"

        lines += [
            f"## {symbol} {title[:80]}",
            "",
            f"- **Result**: {result}",
            f"- **Relative improvement**: {ri:+.2%}",
            f"- **Baseline ppl**: {r.get('baseline_ppl', 'N/A')}",
            f"- **Variant ppl**: {r.get('variant_ppl', 'N/A')}",
        ]
        if r.get("url"):
            lines.append(f"- **Paper**: {r['url']}")
        if r.get("change_description"):
            lines.append(f"- **Change**: {r['change_description']}")
        if r.get("claude_reason"):
            lines.append(f"- **Claude reasoning**: {r['claude_reason']}")
        if r.get("interpretation"):
            lines.append(f"- **Interpretation**: {r['interpretation']}")
        if r.get("old_code") and r.get("new_code"):
            lines += [
                "",
                "**Code change:**",
                "```python",
                f"# Before",
                r["old_code"].strip(),
                "```",
                "```python",
                f"# After",
                r["new_code"].strip(),
                "```",
            ]
        lines += ["", "---", ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[tracker] Task 2 report saved: {path}")
    return path
