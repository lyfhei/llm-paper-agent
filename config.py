import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Paths
ROOT_DIR = Path(__file__).parent
RESULTS_DIR = ROOT_DIR / "results"
REPORTS_DIR = RESULTS_DIR / "reports"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
BASELINE_FILE = RESULTS_DIR / "baseline.json"
MEMORY_FILE = RESULTS_DIR / "agent_memory.json"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR  = RESULTS_DIR / "plots"
MINIMIND_DIR = ROOT_DIR / "minimind"

# API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"

# Task 1
ARXIV_MAX_RESULTS = 30
ARXIV_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]
ARXIV_DAYS_BACK = 90
TOP_K_DEEP_SCORE = 10
LLM_KEYWORDS = [
    "large language model", "llm", "transformer", "fine-tuning",
    "instruction tuning", "rlhf", "reinforcement learning from human feedback",
    "preference optimization", "attention mechanism", "mixture of experts",
    "moe", "retrieval augmented", "rag", "in-context learning",
    "chain of thought", "prompt", "pre-training", "language model",
]

# Task 2
MOCK_EVAL = False          # Set True to use mock evaluator (no real training)
# Training steps per experiment. More steps = lower PPL, more meaningful comparison.
# CPU (no GPU): 2000 steps ≈ 4-5 min. GPU (RTX 5070): 20000 steps ≈ 1 min.
QUICK_TRAIN_STEPS = 2000   # ~5 min on RTX 5070; enough for PPL comparison
PASS_THRESHOLD = 0.02      # relative_improvement > 2% → PASS
CONFIDENCE_THRESHOLD = 0.2 # skip paper if confidence < this
