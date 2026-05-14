import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import BASELINE_FILE, MOCK_EVAL, QUICK_TRAIN_STEPS, MINIMIND_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR

# ── Corpus loading (4-level priority) ─────────────────────────────────────────
# Priority 1: local pretrain_t2t_mini.jsonl in minimind/dataset/
# Priority 2: cached sample in results/corpus_cache.jsonl
# Priority 3: stream first N lines from HuggingFace and cache
# Priority 4: embedded synthetic sentences (fallback, low training signal)

_CORPUS_CACHE = RESULTS_DIR / "corpus_cache.jsonl"

# HuggingFace raw URL for MiniMind's smallest pretraining dataset
_HF_URL = (
    "https://huggingface.co/datasets/jingyaogong/minimind_dataset"
    "/resolve/main/pretrain_t2t_mini.jsonl"
)
_STREAM_N = 5000  # number of texts to stream from HuggingFace and cache (~2.5MB)

# Eval split size (held out from whichever corpus is loaded)
_EVAL_N = 200      # max held-out eval texts (actual count = min(_EVAL_N, 20% of corpus))
_EVAL_SEED = 42


def _read_jsonl(path: Path, max_n: int = 10000) -> List[str]:
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = obj.get("text", "")
                if t and len(t) > 5:
                    texts.append(t)
                    if len(texts) >= max_n:
                        break
            except json.JSONDecodeError:
                continue
    return texts


def _stream_hf(url: str, n: int, timeout: int = 30) -> List[str]:
    """Stream first n valid lines from a JSONL URL without downloading the whole file."""
    import requests
    texts = []
    try:
        print(f"[evaluator] Streaming {n} samples from HuggingFace...")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            buf = b""
            for chunk in r.iter_content(chunk_size=32768):
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        obj = json.loads(line.decode("utf-8", errors="ignore"))
                        t = obj.get("text", "")
                        if t and len(t) > 5:
                            texts.append(t)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                    if len(texts) >= n:
                        return texts
    except Exception as e:
        print(f"[evaluator] HuggingFace stream failed: {e}")
    return texts


def _load_corpus() -> Tuple[List[str], List[str]]:
    """
    Load training corpus with 4-level priority.
    Returns (train_texts, eval_texts) — strictly separated.
    """
    texts = []
    source = "synthetic"

    # --- Priority 1: local file (use as many texts as possible) ---
    local_candidates = [
        MINIMIND_DIR / "dataset" / "pretrain_t2t_mini.jsonl",
        MINIMIND_DIR / "dataset" / "pretrain_t2t.jsonl",
    ]
    for p in local_candidates:
        if p.exists() and p.stat().st_size > 1024:
            texts = _read_jsonl(p, max_n=2_000_000)  # read full dataset
            if texts:
                source = f"local:{p.name}"
                break

    # --- Priority 2: cached sample (must have enough texts) ---
    if not texts and _CORPUS_CACHE.exists() and _CORPUS_CACHE.stat().st_size > 100:
        cached = _read_jsonl(_CORPUS_CACHE, max_n=_STREAM_N + _EVAL_N)
        if len(cached) >= _EVAL_N * 3:   # discard stale small cache
            texts = cached
            source = "cache"
        elif cached:
            print(f"[evaluator] Cache has only {len(cached)} texts (need ≥{_EVAL_N * 3}), re-downloading...")

    # --- Priority 3: stream from HuggingFace ---
    if not texts:
        texts = _stream_hf(_HF_URL, n=_STREAM_N + _EVAL_N)
        if texts:
            source = "huggingface"
            _CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with _CORPUS_CACHE.open("w", encoding="utf-8") as f:
                for t in texts:
                    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            print(f"[evaluator] Cached {len(texts)} samples to {_CORPUS_CACHE}")

    # --- Priority 4: embedded synthetic fallback ---
    if not texts:
        print("[evaluator] WARNING: using synthetic corpus — PPL will be less meaningful.")
        print("[evaluator]   For real data: download minimind/dataset/pretrain_t2t_mini.jsonl")
        texts = _SYNTHETIC_TEXTS

    print(f"[evaluator] Corpus source={source}, total={len(texts)} texts")

    # 80/20 split, capped at _EVAL_N eval texts, minimum 10
    rng = random.Random(_EVAL_SEED)
    shuffled = texts[:]
    rng.shuffle(shuffled)
    n_eval = max(10, min(_EVAL_N, len(shuffled) // 5))
    eval_texts = shuffled[:n_eval]
    train_texts = shuffled[n_eval:]
    print(f"[evaluator] Split: {len(train_texts)} train / {len(eval_texts)} eval")
    return train_texts, eval_texts


# ── Embedded synthetic fallback ────────────────────────────────────────────────
_SYNTHETIC_TEXTS = [
    "人工智能正在改变我们的生活方式。", "语言模型的训练需要大量的计算资源。",
    "深度学习在自然语言处理中取得了重大进展。", "注意力机制是Transformer模型的核心组件。",
    "机器学习算法可以从数据中自动学习规律。", "神经网络由多层神经元组成，能够模拟大脑的工作方式。",
    "预训练模型在下游任务中表现出色。", "自然语言理解是人工智能的重要研究方向。",
    "大型语言模型可以生成连贯的文本内容。", "强化学习可以用于训练智能决策系统。",
    "数据质量对模型性能有着重要影响。", "词向量能够捕捉词语之间的语义关系。",
    "批量归一化有助于加速神经网络的训练。", "梯度消失问题可以通过残差连接来缓解。",
    "多头注意力机制允许模型关注不同的信息。", "位置编码为序列模型提供了位置信息。",
    "知识蒸馏可以将大模型的知识迁移到小模型。", "微调可以使预训练模型适应特定任务。",
    "物理学研究物质、能量以及它们之间的相互作用。", "化学是研究物质组成和变化的科学。",
    "生物学探索生命的起源与演化过程。", "天文学研究宇宙的起源和各种天体的特性。",
    "量子力学描述微观粒子的运动规律。", "相对论改变了人们对时间和空间的认识。",
    "中国有着五千年悠久的历史文明。", "丝绸之路促进了古代东西方的文化交流。",
    "四大发明是中国对世界文明的重要贡献。", "儒家思想深刻影响了中国社会的价值观。",
    "互联网改变了人们获取信息的方式。", "智能手机已成为现代生活不可或缺的工具。",
    "经济全球化促进了各国之间的贸易往来。", "可持续发展是当今社会的重要议题。",
    "大自然的力量既美丽又令人敬畏。", "森林是地球生态系统的重要组成部分。",
    "今天天气不错，我们去公园散步吧。", "健康的饮食习惯对身体有益。",
    "运动可以增强体质，保持身心健康。", "读书是获取知识的重要途径。",
    "问答系统根据问题从知识库中找到答案。", "信息检索从大量文档中找到相关内容。",
    "语义相似度衡量两段文本的意思相近程度。", "文本分类将文本归入预定义的类别。",
    "序列标注为文本中的每个词分配标签。", "关系抽取识别文本中实体之间的关系。",
]


# ── Mock evaluator ─────────────────────────────────────────────────────────────

def _mock_run(base_ppl: float = 45.0) -> Dict:
    ppl = base_ppl + random.uniform(-3.0, 2.0)
    return {
        "ppl": round(ppl, 4),
        "val_loss": round(math.log(ppl), 4),
        "training_time_s": random.randint(10, 30),
        "has_nan": False,
    }


# ── Real evaluator (MiniMind in-process) ─────────────────────────────────────

def _load_minimind_model(tokenizer_cache={}):
    """Import MiniMind model fresh, bypassing Python module cache."""
    if not MINIMIND_DIR.exists():
        raise RuntimeError(
            f"MiniMind not found at {MINIMIND_DIR}. "
            "Run: git clone https://github.com/jingyaogong/minimind minimind"
        )
    minimind_str = str(MINIMIND_DIR)
    if minimind_str not in sys.path:
        sys.path.insert(0, minimind_str)

    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("model.model_minimind") or mod_name == "model.model_minimind":
            del sys.modules[mod_name]

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from transformers import AutoTokenizer

    if "tok" not in tokenizer_cache:
        tokenizer_cache["tok"] = AutoTokenizer.from_pretrained(str(MINIMIND_DIR / "model"))

    return MiniMindConfig, MiniMindForCausalLM, tokenizer_cache["tok"]


_SEQ_LEN    = 340   # matches MiniMind3's max_seq_len
_BATCH_SIZE = 32    # small batch → low activation memory; GPU util stays high via accum


def _encode(text: str, tokenizer, device, max_length: int = _SEQ_LEN):
    """Single-text encode for eval loop (no padding needed)."""
    import torch
    enc = tokenizer(text, add_special_tokens=False,
                    max_length=max_length - 2, truncation=True)
    tokens = [tokenizer.bos_token_id] + enc["input_ids"] + [tokenizer.eos_token_id]
    return torch.tensor([tokens], dtype=torch.long).to(device)


def _encode_batch(texts: list, tokenizer, device):
    """Encode a batch of texts with padding — matches MiniMind's PretrainDataset.
    Returns (input_ids, labels) where padding positions in labels are -100.
    """
    import torch
    pad_id = tokenizer.pad_token_id or 0
    batch_ids = []
    for text in texts:
        enc = tokenizer(text, add_special_tokens=False,
                        max_length=_SEQ_LEN - 2, truncation=True)
        tokens = [tokenizer.bos_token_id] + enc["input_ids"] + [tokenizer.eos_token_id]
        tokens = tokens + [pad_id] * (_SEQ_LEN - len(tokens))
        batch_ids.append(tokens)
    input_ids = torch.tensor(batch_ids, dtype=torch.long).to(device)
    labels = input_ids.clone()
    labels[input_ids == pad_id] = -100   # ignore padding in loss
    return input_ids, labels


def _pretokenize_corpus(texts: list, tokenizer) -> "Tuple[torch.Tensor, torch.Tensor]":
    """Tokenize all texts once → CPU int32 tensors.
    Storing as int32 halves memory vs int64; converted to long when moved to GPU.
    ~100k texts × seq_len=340 ≈ 136 MB, takes a few seconds up-front.
    """
    import torch
    pad_id = tokenizer.pad_token_id or 0
    all_ids: list = []
    _CHUNK = 512   # tokenize in chunks for tqdm accuracy

    try:
        from tqdm import tqdm
        chunks = tqdm(range(0, len(texts), _CHUNK),
                      desc="[evaluator] pre-tokenizing", unit="chunk", leave=False)
    except ImportError:
        chunks = range(0, len(texts), _CHUNK)

    for i in chunks:
        batch = texts[i : i + _CHUNK]
        rows = []
        for text in batch:
            enc = tokenizer(text, add_special_tokens=False,
                            max_length=_SEQ_LEN - 2, truncation=True)
            tokens = ([tokenizer.bos_token_id] + enc["input_ids"]
                      + [tokenizer.eos_token_id])
            tokens = tokens + [pad_id] * (_SEQ_LEN - len(tokens))
            rows.append(tokens)
        all_ids.append(torch.tensor(rows, dtype=torch.int32))

    input_ids = torch.cat(all_ids)          # (N, seq_len) int32 on CPU
    labels = input_ids.clone()
    labels[input_ids == pad_id] = -100
    return input_ids, labels


# Module-level corpus cache so both baseline and variant use identical data
_corpus_cache: Tuple[List[str], List[str]] = ([], [])
_corpus_loaded: bool = False


def _get_corpus() -> Tuple[List[str], List[str]]:
    global _corpus_cache, _corpus_loaded
    if not _corpus_loaded:
        _corpus_cache = _load_corpus()
        _corpus_loaded = True
    return _corpus_cache


_BASE_LR     = 5e-4   # same as MiniMind
_ACCUM_STEPS = 8      # effective batch = 32×8 = 256, exactly matching MiniMind3 (32×8)


def _get_lr(current_eff_step: int, total_eff_steps: int) -> float:
    """MiniMind cosine decay: lr × (0.1 + 0.45 × (1 + cos(π × t/T))).
    Starts at _BASE_LR, ends at 0.1 × _BASE_LR.
    """
    return _BASE_LR * (0.1 + 0.45 * (1 + math.cos(math.pi * current_eff_step / total_eff_steps)))


_PLOT_UPDATE_EVERY = 500  # steps between plot refreshes


def _save_loss_plot(loss_history: list, label: str, plot_path: Path, final: bool = False) -> None:
    """Save (or overwrite) a training loss curve PNG.  Silently skips if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive — works in any terminal / VSCode
        import matplotlib.pyplot as plt

        steps = list(range(1, len(loss_history) + 1))

        # EMA smoothing (alpha=0.98 ≈ window of ~50 steps)
        alpha, ema_val, ema = 0.98, loss_history[0], []
        for l in loss_history:
            ema_val = alpha * ema_val + (1 - alpha) * l
            ema.append(ema_val)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(steps, loss_history, color="steelblue", alpha=0.25, linewidth=0.6, label="raw")
        ax.plot(steps, ema,          color="steelblue", linewidth=1.8,             label="EMA")

        status = "final" if final else f"step {len(steps)}"
        ax.set_title(f"[{label}] training loss  —  {status}", fontsize=11)
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)

        if final:
            print(f"[evaluator] Loss curve saved: {plot_path}")
    except Exception as e:
        print(f"[evaluator] Plot skipped: {e}")


def _run_one(label: str, save_path: "Path | None" = None, plot_path: "Path | None" = None) -> Dict:
    """
    Train a fresh MiniMind from random init, then evaluate on held-out set.
    Both baseline and variant share the same corpus (loaded once per process).
    Matches MiniMind3 training: cosine LR decay + gradient accumulation (_ACCUM_STEPS).
    """
    import torch

    train_texts, eval_texts = _get_corpus()

    MiniMindConfig, MiniMindForCausalLM, tokenizer = _load_minimind_model()

    torch.manual_seed(_EVAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: use_deterministic_algorithms omitted — attention backward is non-deterministic
    # regardless (memory efficient attention), and forcing it causes warnings + no benefit.

    config = MiniMindConfig()  # defaults: hidden_size=768, num_hidden_layers=8 — matches MiniMind3
    model = MiniMindForCausalLM(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()   # clear any fragmentation from previous runs
    model = model.to(device)
    print(f"[evaluator] device={device}  model_device={next(model.parameters()).device}  "
          f"batch={_BATCH_SIZE}  accum={_ACCUM_STEPS}  steps={QUICK_TRAIN_STEPS}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=_BASE_LR)
    # bfloat16 mixed precision — matches MiniMind3, ~2x faster on Ampere+
    _bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    _amp_dtype = torch.bfloat16 if _bf16 else torch.float32

    # On-the-fly tokenization per batch — matches MiniMind3's DataLoader approach.
    # Cost: ~2ms per step (batch=32); GPU step is ~155ms → negligible.
    steps_per_epoch = len(train_texts) // _BATCH_SIZE
    n_steps = steps_per_epoch if QUICK_TRAIN_STEPS <= 0 else min(QUICK_TRAIN_STEPS, steps_per_epoch)
    print(f"[evaluator] corpus_train={len(train_texts)}  steps_per_epoch={steps_per_epoch}  n_steps={n_steps}")

    # Shuffled index pool (CPU list) — slice per step, tokenize that batch on-the-fly
    rng = random.Random(_EVAL_SEED)
    needed = n_steps * _BATCH_SIZE
    pool = list(range(len(train_texts))) * (needed // len(train_texts) + 2)
    rng.shuffle(pool)
    pool = pool[:needed]

    total_eff_steps = max(n_steps // _ACCUM_STEPS, 1)

    has_nan = False
    model.train()
    optimizer.zero_grad()

    # CUDA warmup: pre-compile kernels before tqdm starts so ETA is accurate from step 1
    if device == "cuda":
        print("[evaluator] CUDA warmup (compiling kernels)...")
        _w_ids, _ = _encode_batch(train_texts[:_BATCH_SIZE], tokenizer, device)
        with torch.no_grad(), torch.autocast(device_type=device, dtype=_amp_dtype, enabled=_bf16):
            model(_w_ids)
        del _w_ids
        torch.cuda.synchronize()
        print("[evaluator] CUDA warmup done")

    # tqdm created AFTER warmup so step-1 timing reflects real training speed
    try:
        from tqdm import tqdm
        _pbar = tqdm(range(n_steps), desc=f"[{label}] train", unit="step",
                     dynamic_ncols=True, leave=True)
    except ImportError:
        _pbar = None

    start = time.time()

    recent_losses = []
    all_losses = []   # full history for loss curve
    eff_step = 0

    if plot_path:
        print(f"[evaluator] Loss curve (live): {plot_path}")

    _t_diag = time.time()   # step-timing diagnostic (printed for steps 0-2)
    for step in range(n_steps):
        _t0 = time.time()
        batch_texts = [train_texts[i] for i in pool[step * _BATCH_SIZE : (step + 1) * _BATCH_SIZE]]
        input_ids, labels = _encode_batch(batch_texts, tokenizer, device)
        _t1 = time.time()

        with torch.autocast(device_type=device, dtype=_amp_dtype, enabled=_bf16):
            out = model(input_ids, labels=labels)
        _t2 = time.time()

        if torch.isnan(out.loss):
            has_nan = True
            break

        # Normalize loss for accumulation
        (out.loss / _ACCUM_STEPS).backward()
        _t3 = time.time()

        if step < 3:
            print(f"[diag step {step}] data={(_t1-_t0)*1000:.0f}ms "
                  f"fwd={(_t2-_t1)*1000:.0f}ms bwd={(_t3-_t2)*1000:.0f}ms")

        loss_val = out.loss.item()
        recent_losses.append(loss_val)
        all_losses.append(loss_val)
        if len(recent_losses) > 100:
            recent_losses.pop(0)

        # Optimizer step every _ACCUM_STEPS forward passes
        if (step + 1) % _ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            eff_step += 1
            # MiniMind cosine LR decay
            new_lr = _get_lr(eff_step, total_eff_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

        if _pbar:
            _pbar.update(1)
            if step % 50 == 0:
                avg_l = sum(recent_losses) / len(recent_losses)
                _pbar.set_postfix(loss=f"{avg_l:.3f}",
                                  lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        elif step % 1000 == 0:
            avg_l = sum(recent_losses) / len(recent_losses)
            elapsed_so_far = time.time() - start
            eta = elapsed_so_far / (step + 1) * (n_steps - step - 1)
            print(f"[evaluator] [{label}] step {step}/{n_steps}  "
                  f"loss={avg_l:.3f}  lr={optimizer.param_groups[0]['lr']:.2e}"
                  f"  elapsed={elapsed_so_far:.0f}s  ETA={eta:.0f}s")

        # Periodic plot refresh (overwrite same file — VSCode auto-reloads image)
        if plot_path and all_losses and (step + 1) % _PLOT_UPDATE_EVERY == 0:
            _save_loss_plot(all_losses, label, plot_path)

    if _pbar:
        _pbar.close()

    elapsed = time.time() - start

    # Final loss curve
    if plot_path and all_losses:
        _save_loss_plot(all_losses, label, plot_path, final=True)

    # Save model weights before eval (skip if NaN — weights are unusable)
    if save_path and not has_nan:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[evaluator] Model saved: {save_path.name}")

    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for text in eval_texts:
            input_ids = _encode(text, tokenizer, device)
            if input_ids.shape[1] < 2:
                continue
            with torch.autocast(device_type=device, dtype=_amp_dtype, enabled=_bf16):
                out = model(input_ids, labels=input_ids)
            if not torch.isnan(out.loss):
                total_loss += out.loss.item()
                n += 1

    avg_loss = total_loss / n if n > 0 else float("inf")
    ppl = math.exp(min(avg_loss, 10))

    if ppl > 3000:
        print(f"[evaluator] NOTE: ppl={ppl:.0f} is high — more steps or GPU recommended.")

    print(f"[evaluator] {label}: val_loss={avg_loss:.4f}, ppl={ppl:.4f}, t={elapsed:.1f}s")
    return {
        "ppl": round(ppl, 4),
        "val_loss": round(avg_loss, 4),
        "training_time_s": round(elapsed, 1),
        "has_nan": has_nan,
    }


# ── Load saved model and eval ─────────────────────────────────────────────────

def load_and_eval(model_path: Path) -> Dict:
    """Load a saved model state dict and run eval only (no retraining)."""
    import torch
    MiniMindConfig, MiniMindForCausalLM, tokenizer = _load_minimind_model()
    _, eval_texts = _get_corpus()

    config = MiniMindConfig()  # defaults: hidden_size=768, num_hidden_layers=8 — matches MiniMind3
    model = MiniMindForCausalLM(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    _bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    _amp_dtype = torch.bfloat16 if _bf16 else torch.float32

    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for text in eval_texts:
            input_ids = _encode(text, tokenizer, device)
            if input_ids.shape[1] < 2:
                continue
            with torch.autocast(device_type=device, dtype=_amp_dtype, enabled=_bf16):
                out = model(input_ids, labels=input_ids)
            if not torch.isnan(out.loss):
                total_loss += out.loss.item()
                n += 1

    avg_loss = total_loss / n if n > 0 else float("inf")
    ppl = math.exp(min(avg_loss, 10))
    print(f"[evaluator] Loaded model eval: val_loss={avg_loss:.4f}, ppl={ppl:.4f}")
    return {
        "ppl": round(ppl, 4),
        "val_loss": round(avg_loss, 4),
        "training_time_s": 0,
        "has_nan": False,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def _load_baseline() -> Dict:
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


def run_baseline() -> Dict:
    """
    Run baseline (unpatched model). Cached in BASELINE_FILE.
    Cache is invalidated if QUICK_TRAIN_STEPS changed or ppl is non-finite.
    """
    if BASELINE_FILE.exists():
        cached = _load_baseline()
        ppl = cached.get("baseline_ppl", float("inf"))
        cached_steps = cached.get("quick_train_steps", 0)
        if (
            0 < ppl < float("inf")
            and ppl == ppl  # not NaN
            and cached_steps == QUICK_TRAIN_STEPS
        ):
            print(f"[evaluator] Cached baseline: ppl={ppl:.4f} ({cached_steps} steps)")
            return cached
        print(f"[evaluator] Baseline invalid (ppl={ppl}, steps={cached_steps} vs {QUICK_TRAIN_STEPS}), re-running...")

    print("[evaluator] Running baseline...")
    save_path = MODELS_DIR / "baseline.pt"
    plot_path = PLOTS_DIR / "baseline.png"
    result = _mock_run() if MOCK_EVAL else _run_one("baseline", save_path=save_path, plot_path=plot_path)

    baseline = {
        "baseline_ppl": result["ppl"],
        "baseline_val_loss": result["val_loss"],
        "training_time_s": result["training_time_s"],
        "quick_train_steps": QUICK_TRAIN_STEPS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"[evaluator] Baseline saved: ppl={baseline['baseline_ppl']}")
    return baseline


def run_variant(paper_id: str = None) -> Dict:
    """Run variant (patch already applied on disk).
    If paper_id given and a saved model exists, skip retraining and load it instead.
    Saves model weights to results/models/{paper_id}_variant.pt after new training.
    """
    if paper_id:
        model_path = MODELS_DIR / f"{paper_id}_variant.pt"
        if model_path.exists():
            print(f"[evaluator] Found saved variant for {paper_id}, skipping retrain...")
            result = load_and_eval(model_path)
            return {
                "variant_ppl": result["ppl"],
                "val_loss": result["val_loss"],
                "training_time_s": result["training_time_s"],
                "has_nan": result["has_nan"],
            }

    save_path = (MODELS_DIR / f"{paper_id}_variant.pt") if paper_id else None
    plot_path = (PLOTS_DIR / f"{paper_id}_variant.png") if paper_id else (PLOTS_DIR / "variant.png")
    print(f"[evaluator] Running variant (~{QUICK_TRAIN_STEPS} steps)...")
    result = _mock_run() if MOCK_EVAL else _run_one("variant", save_path=save_path, plot_path=plot_path)
    return {
        "variant_ppl": result["ppl"],
        "val_loss": result["val_loss"],
        "training_time_s": result["training_time_s"],
        "has_nan": result["has_nan"],
    }
