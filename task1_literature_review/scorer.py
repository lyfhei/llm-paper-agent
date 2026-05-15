import json
import time
import anthropic
from typing import List, Dict

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, TOP_K_DEEP_SCORE

SYSTEM_PROMPT = """You are an ML researcher evaluating papers for a small Chinese language model called MiniMind (~26M parameters).

MiniMind architecture:
- Decoder-only transformer with RoPE positional encoding
- RMSNorm for layer normalization
- SwiGLU-style feed-forward (w1, w2, w3 linear layers)
- Standard causal multi-head attention
- Trained on Chinese text, ~500M tokens

Your task: evaluate whether this paper proposes a technique applicable to MiniMind.

Output a JSON object with these exact fields:
- novelty (int 1-10): originality of the contribution
- practicality (int 1-10): how easy to implement on a 26M-param model without large compute
- clarity (int 1-10): how clearly written
- summary (str): one sentence on the key contribution
- target_component (str): which MiniMind component this paper most targets.
  Choose exactly one from: ffn, activation, attention, normalization, positional_encoding, data_pipeline, training_schedule, training_objective, other
- applicability (str): "high" if directly applicable to MiniMind, "medium" if applicable with adaptation, "low" if requires large scale or different architecture
- confidence (float 0.0-1.0): how implementable and worth trying is this technique on MiniMind.
  0.8+ = technique directly matches MiniMind's architecture, easy to implement
  0.6-0.8 = applicable with minor adaptation, clearly worth trying
  0.4-0.6 = indirect relevance, may need significant adaptation
  <0.4 = only for large-scale models or fundamentally different setups

Output ONLY valid JSON, no markdown, no explanation."""


def _coarse_filter(papers: List[Dict], top_k: int) -> List[Dict]:
    # Balance across goals: take top-2 per goal, then fill remaining slots by submission date
    from collections import defaultdict
    by_goal = defaultdict(list)
    for p in papers:
        by_goal[p.get("source_goal", "unknown")].append(p)

    selected, seen = [], set()
    for goal_papers in by_goal.values():
        for p in goal_papers[:2]:
            if p["arxiv_id"] not in seen:
                seen.add(p["arxiv_id"])
                selected.append(p)

    remaining = sorted(
        [p for p in papers if p["arxiv_id"] not in seen],
        key=lambda x: x["submitted_date"],
        reverse=True,
    )
    for p in remaining:
        if len(selected) >= top_k:
            break
        selected.append(p)
        seen.add(p["arxiv_id"])

    return selected[:top_k]


def score_papers(papers: List[Dict]) -> List[Dict]:
    candidates = _coarse_filter(papers, TOP_K_DEEP_SCORE)
    print(f"[scorer] Selected {len(candidates)} papers for deep scoring (balanced across goals)")

    try:
        from task2_experiment import memory as exp_memory
        mem_context = exp_memory.build_context_for_scoring()
    except ImportError:
        mem_context = ""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    scored = []

    for i, paper in enumerate(candidates):
        goal = paper.get("source_goal", "?")
        print(f"[scorer] {i+1}/{len(candidates)} [{goal}] {paper['title'][:50]}...")

        memory_suffix = f"\n\n{mem_context}" if mem_context else ""
        user_msg = (
            f"Improvement goal: {paper.get('goal_description', goal)}\n\n"
            f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            f"{memory_suffix}"
        )
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            if raw and raw[0] != "{":
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start != -1 and end > start:
                    raw = raw[start:end]
            scores = json.loads(raw)
        except Exception as e:
            print(f"  [scorer] Warning: failed for {paper['arxiv_id']}: {e}")
            scores = {
                "novelty": 5, "practicality": 5, "clarity": 5,
                "summary": "Scoring unavailable",
                "target_component": "other",
                "applicability": "low",
                "confidence": 0.0,
            }

        composite = scores["novelty"] * 0.45 + scores["practicality"] * 0.45 + scores["clarity"] * 0.10

        applicability = scores.get("applicability", "low")
        if applicability == "high":
            composite += 0.5
        elif applicability == "medium":
            composite += 0.2

        time.sleep(0.3)

        scored.append({
            **paper,
            "novelty": scores["novelty"],
            "practicality": scores["practicality"],
            "clarity": scores["clarity"],
            "composite_score": round(composite, 2),
            "summary": scores.get("summary", ""),
            "target_component": scores.get("target_component", "other"),
            "applicability": applicability,
            "confidence": scores.get("confidence", 0.0),
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    print(f"[scorer] Done. Top: {scored[0]['title'][:55]} (score={scored[0]['composite_score']})")
    return scored
