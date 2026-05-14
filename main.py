"""
Usage:
  python main.py task1              # Fetch papers, score, generate report
  python main.py task2              # Run agent loop on latest report
  python main.py task2 --mock       # Use mock evaluator (no real training)
  python main.py summary            # Print experiment summary
"""

import sys
import argparse

# Ensure stdout accepts Unicode on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="LLM Paper Agent + MiniMind Experiment Pipeline")
    parser.add_argument("task", choices=["task1", "task2", "summary"], help="Which task to run")
    parser.add_argument("--mock", action="store_true", help="Use mock evaluator for task2 (no real training)")
    parser.add_argument("--max-experiments", type=int, default=3, help="Max experiments for task2 (default: 3)")
    args = parser.parse_args()

    if args.task == "task1":
        from task1_literature_review.agent import run_literature_review

        scored = run_literature_review()
        if scored:
            # Save scored list as JSON for task2 to consume directly
            import json
            from datetime import date
            from config import REPORTS_DIR
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            json_path = REPORTS_DIR / f"{date.today().strftime('%Y-%m-%d')}.json"
            json_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[main] Scored papers saved to {json_path}")

    elif args.task == "task2":
        if args.mock:
            import config
            config.MOCK_EVAL = True
            print("[main] Mock evaluator enabled")

        from task2_experiment.agent_loop import run_agent_loop, _load_latest_report

        papers = _load_latest_report()
        if not papers:
            print("[main] No papers found. Run 'python main.py task1' first.")
            sys.exit(1)

        print(f"[main] Loaded {len(papers)} papers from latest report")
        run_agent_loop(papers, max_experiments=args.max_experiments)

    elif args.task == "summary":
        from task2_experiment.tracker import summary
        summary()


if __name__ == "__main__":
    main()
