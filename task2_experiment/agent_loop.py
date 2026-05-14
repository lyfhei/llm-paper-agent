import ast
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import anthropic
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from config import (
    REPORTS_DIR, CONFIDENCE_THRESHOLD, PASS_THRESHOLD,
    ANTHROPIC_API_KEY, CLAUDE_MODEL, MINIMIND_DIR,
)
from task2_experiment.evaluator import run_baseline, run_variant
from task2_experiment.tracker import save_experiment
from task2_experiment import memory

MODEL_FILE = MINIMIND_DIR / "model" / "model_minimind.py"
MODEL_BAK  = MODEL_FILE.with_suffix(".py.agent_bak")

# ── Relevant code excerpt sent to Claude (skip long generate() method) ─────────
_CODE_EXCERPT_CLASSES = [
    "class MiniMindConfig",
    "class RMSNorm",
    "class FeedForward",
    "class MOEFeedForward",
    "class Attention",
    "class MiniMindBlock",
]

def _get_model_excerpt() -> str:
    """Extract the architecturally-relevant parts of model_minimind.py for Claude."""
    source = MODEL_FILE.read_text(encoding="utf-8")
    lines = source.splitlines()
    keep, in_generate = [], False
    for line in lines:
        if "def generate(" in line:
            in_generate = True
        if in_generate:
            if line.startswith("class ") and "def generate(" not in line:
                in_generate = False
            else:
                continue
        keep.append(line)
    return "\n".join(keep)


# ── Claude: generate patch ─────────────────────────────────────────────────────

_PATCH_SYSTEM = """You are an expert ML engineer implementing research paper techniques on MiniMind, a small Chinese LLM (~26M parameters).

MiniMind uses:
- Decoder-only transformer with RoPE positional encoding (rope_theta=1e6)
- RMSNorm (rms_norm_eps=1e-6)
- SwiGLU feed-forward: down_proj(act_fn(gate_proj(x)) * up_proj(x))
- Grouped Query Attention (8 heads, 4 KV heads)
- All config is in MiniMindConfig.__init__ kwargs

Your job: read the paper and propose ONE minimal, targeted code change to model_minimind.py.

Rules:
1. old_code must be an EXACT verbatim substring of the provided source code
2. new_code must be valid Python — no new external imports beyond what's already imported
3. Change must be small and targeted (one function body, one config default, or one formula)
4. If the paper's technique truly cannot be implemented with a safe minimal change, set can_implement=false
5. CRITICAL: Do NOT add feature flags or boolean parameters that default to False. The change must be ALWAYS ACTIVE in the default code path — modify existing formulas, hyperparameters, or code directly. No new config flags.

YOUR RESPONSE FORMAT:
- Output a single JSON object and NOTHING else
- Start your response immediately with { (no preamble, no explanation, no markdown)
- End your response with } (no trailing text)
- If you cannot implement, still output JSON with can_implement=false"""


def claude_generate_patch(paper: Dict, model_code: str) -> Optional[Dict]:
    """Ask Claude to propose a specific code change based on the paper."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_msg = f"""Paper title: {paper.get('title', '')}
Abstract: {paper.get('abstract', '')}

Here is model_minimind.py (architecturally relevant parts):
```python
{model_code}
```

Based on this paper, propose a minimal code change to model_minimind.py.
Output ONLY the JSON object (start immediately with {{, end with }}, no other text)."""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            system=_PATCH_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        print(f"[agent_loop] Claude raw response (first 300 chars): {raw[:300]!r}")
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        # Find the outermost {...} block (handles preamble text)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end <= start:
            print("[agent_loop] Claude returned no JSON object in response")
            return None
        raw = raw[start:end]
        result = json.loads(raw)
        # Normalize field names (Claude may use 'description' instead of 'change_description')
        if "description" in result and "change_description" not in result:
            result["change_description"] = result.pop("description")
        return result
    except Exception as e:
        print(f"[agent_loop] Claude patch generation failed: {e}")
        return None


# ── Apply / revert Claude-generated change ─────────────────────────────────────

def _validate_change(change: Dict, source: str) -> Optional[str]:
    """Return error string if change is invalid, None if OK."""
    old = change.get("old_code", "")
    new = change.get("new_code", "")
    if not old or not new:
        return "old_code or new_code is empty"
    if old not in source:
        return f"old_code not found verbatim in model_minimind.py"
    patched = source.replace(old, new, 1)
    try:
        ast.parse(patched)
    except SyntaxError as e:
        return f"new_code produces invalid Python: {e}"
    return None


def apply_claude_change(change: Dict) -> bool:
    """Backup model file and apply Claude's change. Returns True on success."""
    source = MODEL_FILE.read_text(encoding="utf-8")
    err = _validate_change(change, source)
    if err:
        print(f"[agent_loop] Patch rejected: {err}")
        return False
    shutil.copy2(MODEL_FILE, MODEL_BAK)
    patched = source.replace(change["old_code"], change["new_code"], 1)
    MODEL_FILE.write_text(patched, encoding="utf-8")
    print(f"[agent_loop] Patch applied: {change.get('change_description', '')}")
    return True


def revert_claude_change() -> None:
    """Restore model file from backup."""
    if MODEL_BAK.exists():
        shutil.copy2(MODEL_BAK, MODEL_FILE)
        MODEL_BAK.unlink()
        print("[agent_loop] Model file reverted.")
    else:
        print("[agent_loop] No backup found to revert.")


# ── Claude: interpret result ───────────────────────────────────────────────────

def claude_interpret_result(paper: Dict, change: Dict, baseline_ppl: float,
                             variant_ppl: float, result: str) -> str:
    """Ask Claude to give a brief interpretation of the experiment result."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    ri = (baseline_ppl - variant_ppl) / baseline_ppl
    prompt = (
        f"Paper: {paper.get('title', '')}\n"
        f"Change: {change.get('change_description', '')}\n"
        f"Baseline ppl: {baseline_ppl:.2f} -> Variant ppl: {variant_ppl:.2f} "
        f"(relative improvement: {ri:+.1%}) -> {result}\n\n"
        "In 2 sentences, explain why this result makes sense given the paper's contribution "
        "and MiniMind's architecture."
    )
    try:
        r = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()
    except Exception:
        return ""


# ── Load report ────────────────────────────────────────────────────────────────

def _load_latest_report() -> list:
    reports = sorted(REPORTS_DIR.glob("*.json"), reverse=True)
    if reports:
        return json.loads(reports[0].read_text(encoding="utf-8"))
    md_reports = sorted(REPORTS_DIR.glob("*.md"), reverse=True)
    if not md_reports:
        raise FileNotFoundError("No Task 1 report found. Run 'python main.py task1' first.")
    return _parse_md_report(md_reports[0])


def _parse_md_report(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    papers = []
    for block in text.split("### #")[1:]:
        lines = block.strip().split("\n")
        title = lines[0].split("—", 1)[-1].strip() if "—" in lines[0] else lines[0]
        arxiv_id = url = ""
        confidence = 0.0
        abstract = ""
        for line in lines:
            if "Link:** " in line:
                url = line.split("Link:** ")[-1].strip()
                arxiv_id = url.rstrip("/").split("/")[-1]
            if "Confidence:**" in line:
                try:
                    confidence = float(line.split("Confidence:** ")[-1].split()[0])
                except (ValueError, IndexError):
                    pass
        papers.append({
            "arxiv_id": arxiv_id, "title": title,
            "url": url, "abstract": abstract,
            "confidence": confidence,
        })
    return papers


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_agent_loop(scored_papers: list, max_experiments: int = 3) -> None:
    print("\n=== Task 2: LLM-in-the-loop Agent ===")

    # Ensure model file is clean before baseline
    revert_claude_change()

    # Run baseline on clean (unpatched) model
    baseline = run_baseline()
    baseline_ppl = baseline["baseline_ppl"]
    print(f"[agent_loop] Baseline ppl={baseline_ppl:.4f}")

    model_code = _get_model_excerpt()
    experiment_count = 0

    for paper in scored_papers:
        if experiment_count >= max_experiments:
            print(f"[agent_loop] Reached max experiments ({max_experiments}). Done.")
            break

        title_short = paper.get("title", "")[:60]
        confidence = paper.get("confidence", 0.0)

        if confidence < CONFIDENCE_THRESHOLD:
            print(f"[agent_loop] Skip '{title_short}': confidence={confidence:.2f}")
            continue

        print(f"\n{'='*66}")
        print(f"  Paper : {title_short}")
        print(f"  Conf  : {confidence:.2f}  |  Goal: {paper.get('source_goal', '')}")
        summary = paper.get("summary", "")
        if summary:
            print(f"  Summary: {summary[:120]}")
        print(f"{'='*66}")

        try:
            ans = input("  Run this experiment? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "y"   # non-interactive (piped), default yes
        if ans in ("n", "no"):
            print("[agent_loop] Skipped by user.")
            continue

        # 1. Claude generates patch
        print("[agent_loop] Asking Claude to propose a code change...")
        change = claude_generate_patch(paper, model_code)

        if change is None or not change.get("can_implement", False):
            reason = change.get("reason", "unknown") if change else "API error"
            print(f"[agent_loop] Claude: cannot implement — {reason}")
            continue

        print(f"[agent_loop] Claude proposes: {change.get('change_description', '')}")
        print(f"[agent_loop] Reason: {change.get('reason', '')}")

        # 2. Apply patch
        if not apply_claude_change(change):
            continue

        experiment_count += 1
        paper_id = paper.get("arxiv_id", f"exp_{experiment_count}")
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        # 3. Run variant (module cache cleared inside run_variant → picks up changed file)
        result = run_variant(paper_id=paper_id)
        variant_ppl = result["variant_ppl"]
        has_nan = result["has_nan"]

        # 4. Decision
        if has_nan:
            decision, relative_improvement = "FAIL", 0.0
            print("[agent_loop] NaN detected -> FAIL")
        else:
            relative_improvement = (baseline_ppl - variant_ppl) / baseline_ppl
            decision = "PASS" if relative_improvement > PASS_THRESHOLD else "FAIL"
            print(
                f"[agent_loop] baseline={baseline_ppl:.4f}, variant={variant_ppl:.4f}, "
                f"improvement={relative_improvement:+.2%} -> {decision}"
            )

        # 5. Always revert so next experiment starts clean
        revert_claude_change()

        # 6. Claude interprets the result
        interpretation = claude_interpret_result(
            paper, change, baseline_ppl, variant_ppl, decision
        )
        if interpretation:
            print(f"[agent_loop] Interpretation: {interpretation}")

        # 7. Record
        record = {
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "url": paper.get("url", ""),
            "experiment_type": "paper-inspired",
            "confidence": confidence,
            "old_code": change.get("old_code", ""),
            "new_code": change.get("new_code", ""),
            "change_description": change.get("change_description", ""),
            "claude_reason": change.get("reason", ""),
            "baseline_ppl": baseline_ppl,
            "variant_ppl": variant_ppl,
            "relative_improvement": round(relative_improvement, 6),
            "result": decision,
            "has_nan": has_nan,
            "training_time_s": result.get("training_time_s", 0),
            "interpretation": interpretation,
            "timestamp": timestamp,
        }
        save_experiment(record)
        memory.update(record)

    print("\n[agent_loop] === Summary ===")
    from task2_experiment.tracker import summary, save_report
    summary()
    report_path = save_report()
    if report_path:
        print(f"[agent_loop] Markdown report: {report_path}")
