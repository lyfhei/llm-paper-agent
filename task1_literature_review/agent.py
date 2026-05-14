from .fetcher import fetch_papers
from .scorer import score_papers
from .report import generate_report, save_report


def _print_summary(papers: list) -> None:
    print("\n" + "=" * 66)
    print(f"  Task 1 Results — {len(papers)} papers scored")
    print("=" * 66)
    print(f"{'#':<3} {'Title':<46} {'Conf':>5} {'Score':>6}")
    print("-" * 66)
    for i, p in enumerate(papers, 1):
        title = p["title"][:45]
        conf  = p.get("confidence", 0.0)
        score = p.get("composite_score", 0.0)
        print(f"{i:<3} {title:<46} {conf:>5.2f} {score:>6.2f}")
    print("=" * 66)

    top = [p for p in papers if p.get("confidence", 0) >= 0.4]
    if top:
        print(f"\nTop {len(top)} candidates for Task 2 (confidence ≥ 0.4):\n")
        for p in top:
            print(f"  [{p.get('confidence',0):.2f}] {p['title'][:65]}")
            print(f"         → {p.get('summary','')[:80]}")
            print(f"         → {p['url']}")
            print()


def run_literature_review() -> list:
    print("=== Task 1: Literature Review ===")
    papers = fetch_papers()
    if not papers:
        print("[agent] No papers found. Check your network or try again later.")
        return []

    scored = score_papers(papers)
    content = generate_report(scored)
    path = save_report(content)

    _print_summary(scored)
    print(f"\n[agent] Full report saved → {path}")
    return scored
