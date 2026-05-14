import json
from pathlib import Path
from typing import Dict

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


def update(record: Dict) -> None:
    """Update agent memory with results from one experiment."""
    data = _load()
    patch_id = record.get("patch_id") or "claude"
    ri = record.get("relative_improvement", 0.0)
    passed = record.get("result") == "PASS"

    # Append to history
    data["patch_history"].append({
        "patch_id": patch_id,
        "paper_id": record.get("paper_id"),
        "paper_title": record.get("title", ""),
        "result": record.get("result"),
        "relative_improvement": ri,
        "future_candidate": passed and ri > 0.05,
        "timestamp": record.get("timestamp", ""),
    })

    # Update stats
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

    _save(data)
    print(f"[memory] Updated: {patch_id}: attempts={stats['attempts']}, passes={stats['passes']}, avg_ri={stats['avg_relative_improvement']:.4f}")


