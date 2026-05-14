import arxiv
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import ARXIV_DAYS_BACK

# Each goal maps to an arXiv query + description.
# Papers fetched per goal, then deduplicated.
IMPROVEMENT_GOALS = {
    "architecture": {
        "query": 'abs:"feed-forward" AND (abs:"SwiGLU" OR abs:"GeGLU" OR abs:"gated" OR abs:"activation function") AND (abs:"language model" OR abs:"transformer")',
        "description": "Better FFN / activation architecture for transformers",
        "per_goal_limit": 8,
    },
    "training_efficiency": {
        "query": '(abs:"curriculum learning" OR abs:"progressive training" OR abs:"sample ordering" OR abs:"data scheduling") AND (abs:"language model" OR abs:"LLM" OR abs:"transformer")',
        "description": "More efficient training strategies",
        "per_goal_limit": 8,
    },
    "data_quality": {
        "query": '(abs:"data quality" OR abs:"data filtering" OR abs:"data curation" OR abs:"data selection") AND (abs:"pretraining" OR abs:"language model" OR abs:"LLM")',
        "description": "Better training data selection and filtering",
        "per_goal_limit": 8,
    },
    "training_objective": {
        "query": '(abs:"knowledge distillation" OR abs:"training objective" OR abs:"loss function") AND (abs:"language model" OR abs:"small model" OR abs:"LLM")',
        "description": "Better training objectives and loss functions",
        "per_goal_limit": 8,
    },
}


def _fetch_for_goal(goal_name: str, goal_cfg: dict, days_back: int) -> List[Dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    client = arxiv.Client(page_size=50, delay_seconds=5, num_retries=5)
    search = arxiv.Search(
        query=goal_cfg["query"],
        max_results=goal_cfg["per_goal_limit"] * 3,
        sort_by=arxiv.SortCriterion.Relevance,  # arXiv relevance score as hotness proxy
    )

    # Retry with exponential backoff on 429
    for attempt in range(4):
        try:
            papers = []
            for result in client.results(search):
                if result.published < cutoff:
                    continue
                papers.append({
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "authors": [a.name for a in result.authors[:5]],
                    "abstract": result.summary.replace("\n", " "),
                    "url": result.entry_id,
                    "submitted_date": result.published.strftime("%Y-%m-%d"),
                    "categories": result.categories,
                    "source_goal": goal_name,
                    "goal_description": goal_cfg["description"],
                })
                if len(papers) >= goal_cfg["per_goal_limit"]:
                    break
            print(f"[fetcher] Goal '{goal_name}': {len(papers)} papers")
            return papers
        except Exception as e:
            wait = 15 * (2 ** attempt)  # 15s, 30s, 60s, 120s
            print(f"[fetcher] Goal '{goal_name}' error (attempt {attempt+1}): {e}")
            if attempt < 3:
                print(f"[fetcher] Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"[fetcher] Skipping goal '{goal_name}' after 4 attempts.")
                return []


def fetch_papers(days_back: int = ARXIV_DAYS_BACK) -> List[Dict]:
    seen_ids = set()
    all_papers = []

    for i, (goal_name, goal_cfg) in enumerate(IMPROVEMENT_GOALS.items()):
        if i > 0:
            print("[fetcher] Waiting 5s between goals (arXiv rate limit)...")
            time.sleep(5)
        papers = _fetch_for_goal(goal_name, goal_cfg, days_back)
        for p in papers:
            if p["arxiv_id"] not in seen_ids:
                seen_ids.add(p["arxiv_id"])
                all_papers.append(p)

    print(f"[fetcher] Total unique papers: {len(all_papers)} (across {len(IMPROVEMENT_GOALS)} goals)")
    return all_papers
