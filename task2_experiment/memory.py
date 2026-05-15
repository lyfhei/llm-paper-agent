import json
import time
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import MEMORY_FILE


def _load() -> Dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return {"patch_history": [], "patch_stats": {}}


def _save(data: Dict) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _update_technique_registry(data: Dict, record: Dict) -> None:
    technique = record.get("technique_category") or record.get("target_component", "other")
    if not technique:
        technique = "other"
    ri = record.get("relative_improvement", 0.0)
    passed = record.get("result") == "PASS"
    paper_id = record.get("paper_id", "")

    registry = data.setdefault("technique_registry", {})
    entry = registry.setdefault(technique, {
        "attempts": 0, "passes": 0,
        "best_ri": None, "worst_ri": None, "avg_ri": 0.0,
        "paper_ids": [],
    })

    prev_avg = entry["avg_ri"]
    prev_n = entry["attempts"]
    entry["attempts"] += 1
    if passed:
        entry["passes"] += 1
    entry["avg_ri"] = round((prev_avg * prev_n + ri) / entry["attempts"], 6)
    entry["best_ri"] = ri if entry["best_ri"] is None else max(entry["best_ri"], ri)
    entry["worst_ri"] = ri if entry["worst_ri"] is None else min(entry["worst_ri"], ri)
    if paper_id and paper_id not in entry["paper_ids"]:
        entry["paper_ids"].append(paper_id)


def update(record: Dict) -> None:
    data = _load()
    patch_id = record.get("patch_id") or "claude"
    ri = record.get("relative_improvement", 0.0)
    passed = record.get("result") == "PASS"

    data["patch_history"].append({
        "patch_id": patch_id,
        "paper_id": record.get("paper_id"),
        "paper_title": record.get("title", ""),
        "result": record.get("result"),
        "relative_improvement": ri,
        "future_candidate": passed and ri > 0.05,
        "timestamp": record.get("timestamp", ""),
    })

    stats = data["patch_stats"].setdefault(patch_id, {
        "attempts": 0, "passes": 0, "avg_relative_improvement": 0.0
    })
    prev_avg = stats["avg_relative_improvement"]
    prev_attempts = stats["attempts"]
    stats["attempts"] += 1
    if passed:
        stats["passes"] += 1
    stats["avg_relative_improvement"] = round(
        (prev_avg * prev_attempts + ri) / stats["attempts"], 4
    )

    _update_technique_registry(data, record)
    _save(data)
    print(f"[memory] Updated: {patch_id}: attempts={stats['attempts']}, passes={stats['passes']}, avg_ri={stats['avg_relative_improvement']:.4f}")


def update_with_analysis(record: Dict, analysis: Dict) -> None:
    data = _load()
    technique = analysis.get("technique_category", record.get("target_component", "other"))
    entry = {
        "paper_id": record.get("paper_id", ""),
        "paper_title": record.get("title", ""),
        "timestamp": record.get("timestamp", ""),
        "technique_category": technique,
        "result": record.get("result", ""),
        "relative_improvement": record.get("relative_improvement", 0.0),
        "change_description": record.get("change_description", ""),
        "what_failed": analysis.get("what_failed", ""),
        "root_cause": analysis.get("root_cause", ""),
        "avoid_pattern": analysis.get("avoid_pattern", ""),
        "transferable_lesson": analysis.get("transferable_lesson", ""),
    }
    data.setdefault("failure_analyses", []).append(entry)
    _update_technique_registry(data, {**record, "technique_category": technique})
    _save(data)


def update_memory_summary(summary: Dict) -> None:
    data = _load()
    history = data.get("patch_history", [])
    total = len(history)
    passes = sum(1 for r in history if r.get("result") == "PASS")
    data["memory_summary"] = {
        **summary,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_experiments": total,
        "pass_rate": round(passes / total, 4) if total else 0.0,
    }
    _save(data)


def build_context_for_patch() -> str:
    data = _load()
    history = data.get("patch_history", [])
    if not history:
        return ""

    total = len(history)
    passes = sum(1 for r in history if r.get("result") == "PASS")
    avg_ri = sum(r.get("relative_improvement", 0) for r in history) / total

    lines = [
        f"=== EXPERIMENT MEMORY ({total} past experiments) ===",
        f"OVERALL: {passes}/{total} PASS ({passes/total:.0%}), avg improvement: {avg_ri:+.1%}",
        "",
        "TECHNIQUE RESULTS:",
    ]

    registry = data.get("technique_registry", {})
    for tech, entry in sorted(registry.items(), key=lambda x: x[1].get("avg_ri", 0), reverse=True):
        a = entry["attempts"]
        p = entry["passes"]
        best = entry.get("best_ri")
        best_str = f"{best:+.1%}" if best is not None else "N/A"
        if p > 0:
            label = "SOME SUCCESSES"
        elif entry.get("worst_ri", 0) < -0.5:
            label = "CRITICAL RISK"
        else:
            label = "AVOID"
        lines.append(f"- {tech}: {p} PASS / {a} attempts, best {best_str} — {label}")

    analyses = data.get("failure_analyses", [])
    if analyses:
        lines += ["", "RECENT FAILURE LESSONS:"]
        for i, fa in enumerate(analyses[-3:], 1):
            lines.append(f"{i}. [{fa.get('technique_category', 'other')}] {fa.get('paper_title', '')[:50]}")
            if fa.get("what_failed"):
                lines.append(f"   What failed: {fa['what_failed'][:120]}")
            if fa.get("root_cause"):
                lines.append(f"   Root cause: {fa['root_cause'][:120]}")
            if fa.get("avoid_pattern"):
                lines.append(f"   Avoid: {fa['avoid_pattern'][:100]}")

    summary = data.get("memory_summary", {})
    if summary.get("synthesis"):
        lines += ["", f"SYNTHESIS: {summary['synthesis']}"]

    lines.append("=== END MEMORY ===")
    return "\n".join(lines)


def build_context_for_scoring() -> str:
    data = _load()
    history = data.get("patch_history", [])
    if not history:
        return ""

    registry = data.get("technique_registry", {})
    successes = {t: e for t, e in registry.items() if e.get("passes", 0) > 0}
    failures = {t: e for t, e in registry.items() if e.get("passes", 0) == 0}

    lines = ["=== PAST EXPERIMENT RESULTS (use to calibrate confidence) ==="]

    if successes:
        lines.append("Successes (PASS):")
        for tech, e in successes.items():
            lines.append(f"- {tech}: {e['passes']}/{e['attempts']} passed, best {e['best_ri']:+.1%}")

    if failures:
        lines.append("Failures to note:")
        for tech, e in sorted(failures.items(), key=lambda x: x[1].get("avg_ri", 0)):
            lines.append(f"- {tech}: 0/{e['attempts']} passed, avg {e['avg_ri']:+.1%}")

    avoid = [t for t, e in failures.items() if e.get("worst_ri", 0) < -0.5]
    boost = [t for t in successes]

    parts = []
    if avoid:
        parts.append(f"Lower confidence (0.15-0.25) for papers proposing: {', '.join(avoid)}")
    if boost:
        parts.append(f"Raise confidence (0.05-0.10) for papers proposing: {', '.join(boost)}")
    if parts:
        lines += ["", "CALIBRATION GUIDANCE:"] + [f"- {p}" for p in parts]

    lines.append("=== END PAST RESULTS ===")
    return "\n".join(lines)
