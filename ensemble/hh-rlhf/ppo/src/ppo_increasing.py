#!/usr/bin/env python
import os, json, gc, yaml, argparse, logging, time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import accelerate
from accelerate import Accelerator
import mlflow

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import wandb


# ──────────────────────────────────────────────────────────────────────────────
# monkey-patch Accelerator.__init__ to keep dispatch_batches arg
# ──────────────────────────────────────────────────────────────────────────────
_orig_accel_init = accelerate.Accelerator.__init__
def _patched_accel_init(self, *args, dispatch_batches=None, **kwargs):
    return _orig_accel_init(self, *args, **kwargs)
accelerate.Accelerator.__init__ = _patched_accel_init


# ──────────────────────────────────────────────────────────────────────────────
# (1) ONLY CHANGE: patch TRL logprob computation to avoid full [B,T,V] log_softmax
# ──────────────────────────────────────────────────────────────────────────────
import trl.core as trl_core
import trl.trainer.ppo_trainer as trl_ppo_trainer_mod

def _logprobs_from_logits_no_full_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, T, V], labels: [B, T]
    returns: [B, T] token logprobs without materializing [B,T,V] log_softmax.
    log p(label) = logits[label] - logsumexp(logits)
    """
    tok_logits = logits.gather(dim=2, index=labels.unsqueeze(2)).squeeze(2)  # [B, T]
    lse = torch.logsumexp(logits, dim=2)                                     # [B, T]
    return tok_logits - lse

trl_core.logprobs_from_logits = _logprobs_from_logits_no_full_softmax
trl_ppo_trainer_mod.logprobs_from_logits = _logprobs_from_logits_no_full_softmax


# ──────────────────────────────────────────────────────────────────────────────
# HF caches: node-local by default (outputs/logs stay on /efs)
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf/datasets")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ──────────────────────────────────────────────────────────────────────────────
# memory behavior (helps fragmentation / allocator)
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128",
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ──────────────────────────────────────────────────────────────────────────────
# bf16 autocast helper (A100 supports bf16)
# ──────────────────────────────────────────────────────────────────────────────
def bf16_autocast_if_available(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _cleanup_cuda(aggressive: bool = False):
    if aggressive:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# gather_object replacement (older accelerate has no Accelerator.gather_object)
# ──────────────────────────────────────────────────────────────────────────────
def gather_object_all_ranks(obj, accelerator: Accelerator):
    """
    Returns list[world_size] with obj from each rank.
    Works even if accelerate lacks gather_object.
    """
    if dist.is_available() and dist.is_initialized():
        out = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(out, obj)
        return out
    return [obj]


# ──────────────────────────────────────────────────────────────────────────────
# utilities
# ──────────────────────────────────────────────────────────────────────────────
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ppo_training_log_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def _write_done(path: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("ok\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _model_done_file(model_dir: str) -> str:
    return os.path.join(model_dir, "_DONE")


def _model_is_done(model_dir: str) -> bool:
    return os.path.isdir(model_dir) and os.path.exists(_model_done_file(model_dir))


def _sample_k_unique_indices(n: int, k: int, seed: int) -> List[int]:
    n = int(n)
    k = int(k)
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(int(seed))
    return rng.choice(n, size=k, replace=False).tolist()


def wait_for_all_reward_models(model_root: str, m: int, *, sleep_s: float = 5.0, logger=None):
    """
    IMPORTANT: no collectives here. Pure filesystem polling to avoid NCCL timeouts
    when some ranks have no RM training work and would otherwise block in barriers.
    """
    model_dirs = [os.path.join(model_root, f"rm_{j}") for j in range(int(m))]
    while True:
        missing = [d for d in model_dirs if not _model_is_done(d)]
        if not missing:
            return
        if logger is not None:
            logger.info(f"[sync] waiting for reward models to finish. missing={len(missing)}")
        time.sleep(float(sleep_s))


# ──────────────────────────────────────────────────────────────────────────────
# PPO gradient checkpointing helper (NEW)
# ──────────────────────────────────────────────────────────────────────────────
def enable_gradient_checkpointing_for_ppo(model, *, logger=None, tag: str = "policy"):
    """
    TRL's AutoModelForCausalLMWithValueHead wraps an underlying LM (often at .pretrained_model).
    We enable checkpointing wherever supported and force use_cache=False (required).
    """
    enabled_any = False

    # 1) wrapper-level
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            enabled_any = True
        except Exception:
            pass

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # 2) underlying LM(s)
    for attr in ("pretrained_model", "base_model", "model", "transformer"):
        if not hasattr(model, attr):
            continue
        sub = getattr(model, attr)
        if sub is None:
            continue

        if hasattr(sub, "gradient_checkpointing_enable"):
            try:
                sub.gradient_checkpointing_enable()
                enabled_any = True
            except Exception:
                pass

        if hasattr(sub, "config") and hasattr(sub.config, "use_cache"):
            sub.config.use_cache = False

    if logger is not None:
        logger.info(f"[ppo] gradient_checkpointing({tag}) enabled={enabled_any}; use_cache forced False")


# ──────────────────────────────────────────────────────────────────────────────
# HH-RLHF dataset loading (EXACTLY like your second code)
# ──────────────────────────────────────────────────────────────────────────────
def build_dataset_hh(
    dataset_name="Anthropic/hh-rlhf",
    model_name=None,
    input_min_text_length=2,
    input_max_text_length=8,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")

    def extract_prompt(conv: str) -> str:
        idx = conv.rfind("Assistant:")
        return conv[:idx].strip() if idx != -1 else conv.strip()

    def tokenize(sample):
        ids = tokenizer.encode(
            extract_prompt(sample["chosen"]),
            padding=False,
            truncation=True,
            max_length=256,
        )
        sample["input_ids"] = ids
        sample["query"] = tokenizer.decode(ids, skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def drop_to_ppo_columns(ds):
    keep = {"input_ids", "query"}
    drop = [c for c in ds.column_names if c not in keep]
    if len(drop) > 0:
        ds = ds.remove_columns(drop)
    ds.set_format(type="torch", columns=["input_ids", "query"])
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap ensemble utilities (GPT2 reward model)
# ──────────────────────────────────────────────────────────────────────────────
class _PairwiseHHDataset(Dataset):
    def __init__(self, ds):
        if "chosen" not in ds.column_names or "rejected" not in ds.column_names:
            raise ValueError(f"HH RM training expects columns chosen/rejected. Found: {ds.column_names}")
        self._chosen = ds["chosen"]
        self._rejected = ds["rejected"]

    def __len__(self):
        return len(self._chosen)

    def __getitem__(self, i):
        return {"chosen": self._chosen[i], "rejected": self._rejected[i]}


_RM_TOK = None
_RM_MAXLEN = None

def _rm_worker_init(worker_id: int, base_model_name: str, max_length: int):
    global _RM_TOK, _RM_MAXLEN
    _RM_TOK = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, local_files_only=True)
    if _RM_TOK.pad_token_id is None:
        _RM_TOK.pad_token_id = _RM_TOK.eos_token_id
    _RM_MAXLEN = int(max_length)

def rm_pairwise_collate_worker(batch: List[Dict[str, Any]]):
    global _RM_TOK, _RM_MAXLEN
    if _RM_TOK is None or _RM_MAXLEN is None:
        raise RuntimeError("RM tokenizer not initialized in worker. Check worker_init_fn.")
    chosen = [b["chosen"] for b in batch]
    rejected = [b["rejected"] for b in batch]
    enc_c = _RM_TOK(chosen, return_tensors="pt", truncation=True, padding=True, max_length=_RM_MAXLEN)
    enc_r = _RM_TOK(rejected, return_tensors="pt", truncation=True, padding=True, max_length=_RM_MAXLEN)
    return {
        "c_input_ids": enc_c["input_ids"],
        "c_attention_mask": enc_c.get("attention_mask", None),
        "r_input_ids": enc_r["input_ids"],
        "r_attention_mask": enc_r.get("attention_mask", None),
    }

def rm_pairwise_collate_mainproc(batch: List[Dict[str, Any]], tok, max_length: int):
    chosen = [b["chosen"] for b in batch]
    rejected = [b["rejected"] for b in batch]
    enc_c = tok(chosen, return_tensors="pt", truncation=True, padding=True, max_length=int(max_length))
    enc_r = tok(rejected, return_tensors="pt", truncation=True, padding=True, max_length=int(max_length))
    return {
        "c_input_ids": enc_c["input_ids"],
        "c_attention_mask": enc_c.get("attention_mask", None),
        "r_input_ids": enc_r["input_ids"],
        "r_attention_mask": enc_r.get("attention_mask", None),
    }


def _pairwise_bt_loss(chosen_score: torch.Tensor, rejected_score: torch.Tensor) -> torch.Tensor:
    diff = chosen_score - rejected_score
    return F.softplus(-diff).mean()


def train_reward_model_gpt_pairwise(
    ds_shard,
    *,
    base_model_name: str,
    output_dir: str,
    device: torch.device,
    lr: float,
    batch_size: int,
    epochs: int,
    max_length: int,
    grad_accum_steps: int = 1,
    max_steps: int | None = None,
    seed: int = 42,
    num_workers: int = 0,
    gradient_checkpointing: bool = True,
    logger=None,
):
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, trust_remote_code=True).to(device)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id

    model.train()

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    dataset = _PairwiseHHDataset(ds_shard)

    nw = int(max(0, num_workers))
    if nw > 0:
        collate_fn = rm_pairwise_collate_worker
        worker_init_fn = partial(_rm_worker_init, base_model_name=base_model_name, max_length=int(max_length))
        pin_memory = True
        prefetch_factor = 2
    else:
        collate_fn = partial(rm_pairwise_collate_mainproc, tok=tok, max_length=int(max_length))
        worker_init_fn = None
        pin_memory = False
        prefetch_factor = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        drop_last=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    for ep in range(epochs):
        for batch in loader:
            c_input_ids = batch["c_input_ids"].to(device, non_blocking=True)
            r_input_ids = batch["r_input_ids"].to(device, non_blocking=True)

            c_attn = batch["c_attention_mask"]
            r_attn = batch["r_attention_mask"]
            if c_attn is not None:
                c_attn = c_attn.to(device, non_blocking=True)
            if r_attn is not None:
                r_attn = r_attn.to(device, non_blocking=True)

            with bf16_autocast_if_available(device):
                c_out = model(input_ids=c_input_ids, attention_mask=c_attn)
                r_out = model(input_ids=r_input_ids, attention_mask=r_attn)
                c_score = c_out.logits.squeeze(-1)
                r_score = r_out.logits.squeeze(-1)
                loss = _pairwise_bt_loss(c_score.float(), r_score.float())
                loss = loss / max(1, grad_accum_steps)

            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if logger is not None and step % 50 == 0:
                logger.info(f"[RM train GPT] out={output_dir} step={step} loss={float(loss.item()):.6f}")

            del c_input_ids, r_input_ids, c_attn, r_attn, c_out, r_out, c_score, r_score, loss
            if step % 100 == 0:
                _cleanup_cuda(aggressive=False)

            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    model.eval()
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    _write_done(_model_done_file(output_dir))

    del model, tok
    _cleanup_cuda(aggressive=True)
    return output_dir


def consensus_collate_idx_hh(examples: List[Dict[str, Any]]):
    return {
        "idx": [int(e["__idx__"]) for e in examples],
        "chosen": [e["chosen"] for e in examples],
        "rejected": [e["rejected"] for e in examples],
    }


@torch.no_grad()
def compute_consensus_scores_parallel_gpt(
    ds,
    model_dirs: List[str],
    *,
    gamma: float,
    device: torch.device,
    batch_size: int,
    max_length: int,
    accelerator: Accelerator,
    num_workers: int = 0,
    logger=None,
) -> List[float] | None:
    needed = {"chosen", "rejected"}
    for c in needed:
        if c not in ds.column_names:
            raise ValueError(f"Consensus scoring needs column '{c}', found columns={ds.column_names}")

    indexed = ds.add_column("__idx__", list(range(len(ds))))

    sampler = DistributedSampler(
        indexed,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
    )

    nw = int(max(0, num_workers))
    loader = DataLoader(
        indexed,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=consensus_collate_idx_hh,
        num_workers=nw,
        pin_memory=(nw > 0),
        persistent_workers=False,
        prefetch_factor=2 if nw > 0 else None,
        drop_last=False,
    )

    local_idx: List[int] = []
    local_chosen: List[str] = []
    local_rejected: List[str] = []

    pbar0 = tqdm(loader, desc="Consensus materialize (GPT RM)", disable=not accelerator.is_local_main_process)
    for batch in pbar0:
        local_idx.extend(batch["idx"])
        local_chosen.extend(batch["chosen"])
        local_rejected.extend(batch["rejected"])

    L = len(local_idx)
    if L == 0:
        accelerator.wait_for_everyone()
        _ = gather_object_all_ranks([], accelerator)
        if not accelerator.is_main_process:
            return None
        return [float("-inf")] * len(ds)

    margins = np.empty((len(model_dirs), L), dtype=np.float32)

    for j, md in enumerate(model_dirs):
        tok = AutoTokenizer.from_pretrained(md, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

        model = AutoModelForSequenceClassification.from_pretrained(md, trust_remote_code=True).eval().to(device)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tok.pad_token_id

        for start in range(0, L, batch_size):
            end = min(L, start + batch_size)
            chosen = local_chosen[start:end]
            rejected = local_rejected[start:end]

            enc_c = tok(chosen, return_tensors="pt", truncation=True, padding=True, max_length=int(max_length)).to(device)
            enc_r = tok(rejected, return_tensors="pt", truncation=True, padding=True, max_length=int(max_length)).to(device)

            with bf16_autocast_if_available(device):
                c_out = model(**enc_c)
                r_out = model(**enc_r)
                diff = (c_out.logits.squeeze(-1) - r_out.logits.squeeze(-1)).float().detach().cpu().numpy()

            margins[j, start:end] = diff

            del enc_c, enc_r, c_out, r_out, diff
            if (start // batch_size) % 20 == 0:
                _cleanup_cuda(aggressive=False)

        del model, tok
        _cleanup_cuda(aggressive=True)

    mu = margins.mean(axis=0)
    sigma = margins.std(axis=0, ddof=1) if len(model_dirs) > 1 else np.zeros_like(mu)
    C = mu - float(gamma) * sigma

    local_pairs: List[Tuple[int, float]] = [(int(i), float(c)) for i, c in zip(local_idx, C)]

    accelerator.wait_for_everyone()
    gathered = gather_object_all_ranks(local_pairs, accelerator)

    _cleanup_cuda(aggressive=True)

    if not accelerator.is_main_process:
        return None

    flat: List[Tuple[int, float]] = []
    for item in gathered:
        if isinstance(item, list):
            flat.extend(item)

    N = len(ds)
    out = [float("nan")] * N
    for i, c in flat:
        out[i] = c
    for i in range(N):
        if not np.isfinite(out[i]):
            out[i] = float("-inf")

    if logger is not None:
        logger.info(f"[bootstrap ensemble] consensus scores computed for N={N} (GPT RM)")
    return out


@torch.no_grad()
def compute_val_reward(
    ppo_trainer,
    val_dataloader,
    tokenizer,
    rm_model,
    rm_tok,
    accelerator,
    generation_kwargs,
    batch_size: int = 16,
):
    device = next(rm_model.parameters()).device
    rm_model.eval()

    local_sum = torch.tensor(0.0, device=device)
    local_count = torch.tensor(0.0, device=device)

    progress_bar = tqdm(val_dataloader, desc="Validation (reward)", disable=not accelerator.is_local_main_process)

    for batch in progress_bar:
        prompts = batch["query"]
        query_tensors = batch["input_ids"]
        input_lengths = [len(x) for x in query_tensors]

        gen_outputs = ppo_trainer.generate(query_tensor=query_tensors, batch_size=batch_size, **generation_kwargs)

        response_tensors = [out_ids[input_len:] for out_ids, input_len in zip(gen_outputs, input_lengths)]
        generated_texts = [tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in response_tensors]

        full_texts = [p + " " + g for p, g in zip(prompts, generated_texts)]
        rm_inputs = rm_tok(full_texts, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)

        with bf16_autocast_if_available(device):
            logits = rm_model(**rm_inputs).logits.squeeze(-1)

        local_sum += logits.float().sum()
        local_count += torch.tensor(logits.numel(), device=device, dtype=torch.float32)

        del rm_inputs, logits, full_texts, generated_texts, response_tensors, gen_outputs
        _cleanup_cuda(aggressive=False)

    accelerator.wait_for_everyone()
    global_sum = accelerator.reduce(local_sum, reduction="sum")
    global_count = accelerator.reduce(local_count, reduction="sum")

    avg_reward = (global_sum / (global_count + 1e-8)).item()
    if accelerator.is_main_process:
        print(f"[VAL] avg_reward = {avg_reward:.4f}")
    return avg_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="../config/config.yaml", type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)

    log_dir = cfg.get("logging", {}).get("dir", "/efs/rohandeb/logs/ensemble/qwen/hh-rlhf/ppo_gpt_rm_ensemble")
    logger = setup_logging(log_dir)
    logger.info("Starting PPO training with bootstrap ensemble curriculum (HH-RLHF, GPT2 RM ensemble)")
    logger.info(json.dumps(cfg, indent=2, default=str))

    accelerator = Accelerator()

    dataset_name = cfg["dataset"]["name"]
    model_name = cfg["model_name"]
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])
    ppo_epochs = int(cfg["training"]["num_epochs"])
    init_kl_coef = float(cfg["ppo"]["init_kl_coef"])
    target_kl = float(cfg["ppo"]["target_kl"])
    max_response_length = int(cfg["dataset"]["max_response_length"])
    mini_batch_size = int(cfg["ppo"]["mini_batch_size"])

    # NEW: PPO gradient checkpointing toggle (defaults True)
    ppo_grad_ckpt = bool(cfg.get("training", {}).get("gradient_checkpointing", True))

    output_dir = cfg["checkpoints"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    default_run_name = f"ppo_ensemble_bs{batch_size}_lr{lr:.1e}_kl{init_kl_coef}"
    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", True))
    wandb_run_name = wandb_cfg.get("run_name", default_run_name)

    if cfg.get("mlflow", {}).get("enabled", True):
        mlflow.set_tracking_uri(cfg["mlflow"].get("tracking_uri", "https://mlflow.rlscience.scot.amazon.dev"))
        mlflow.set_experiment(cfg["mlflow"].get("experiment", "ppo-ensemble-curr"))
        if accelerator.is_main_process:
            mlflow.start_run(run_name=cfg["mlflow"].get("run_name", wandb_run_name))

    if accelerator.is_main_process and wandb_enabled:
        wandb.login(key=wandb_cfg["key"])
        wandb_config = {
            **cfg,
            "training/batch_size": batch_size,
            "training/learning_rate": lr,
            "ppo/init_kl_coef": init_kl_coef,
            "training/gradient_checkpointing": ppo_grad_ckpt,
        }
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            name=wandb_run_name,
            config=wandb_config,
        )

    raw_ds = build_dataset_hh(dataset_name=dataset_name, model_name=model_name)
    logger.info(f"Loaded raw dataset with {len(raw_ds)} examples")

    if cfg["dataset"].get("max_samples"):
        raw_ds = raw_ds.select(range(int(cfg["dataset"]["max_samples"])))
        logger.info(f"Subsampled to {len(raw_ds)} examples via max_samples")

    split_seed = int(cfg["dataset"].get("split_seed", 42))
    split = raw_ds.train_test_split(test_size=0.10, seed=split_seed)
    train_ds, eval_ds = split["train"], split["test"]

    be_cfg = cfg.get("bootstrap_ensemble", {})
    be_enabled = bool(be_cfg.get("enabled", True))

    if be_enabled:
        m = int(be_cfg.get("m", 5))
        gamma = float(be_cfg.get("gamma", 0.0))
        shard_seed = int(be_cfg.get("shard_seed", split_seed))

        base_rm = be_cfg.get("base_model_name", "Ray2333/gpt2-large-helpful-reward_model")
        rm_k = int(be_cfg.get("rm_train_samples_per_model", 10_000))

        cache_dir = be_cfg.get("cache_dir", os.path.join(log_dir, "bootstrap_ensemble_cache_gpt_rm"))
        reuse_cache = bool(be_cfg.get("reuse_cache", True))
        os.makedirs(cache_dir, exist_ok=True)

        order_path = os.path.join(cache_dir, f"sorted_indices_m{m}_gamma{gamma}_seed{shard_seed}.json")
        model_root = os.path.join(cache_dir, f"reward_models_m{m}_seed{shard_seed}_k{rm_k}")
        os.makedirs(model_root, exist_ok=True)

        if accelerator.is_main_process:
            _ = AutoTokenizer.from_pretrained(base_rm, trust_remote_code=True)
            _ = AutoModelForSequenceClassification.from_pretrained(base_rm, trust_remote_code=True)
        accelerator.wait_for_everyone()

        if reuse_cache and os.path.exists(order_path):
            if accelerator.is_main_process:
                logger.info(f"[bootstrap ensemble] Found cached ordering at {order_path}; skipping RM training/scoring.")
            accelerator.wait_for_everyone()
        else:
            rm_train = be_cfg.get("rm_train", {})
            rm_lr = float(rm_train.get("learning_rate", 1e-5))
            rm_bs = int(rm_train.get("batch_size", 8))
            rm_epochs = int(rm_train.get("epochs", 1))
            rm_max_len = int(rm_train.get("max_length", 1024))
            rm_grad_accum = int(rm_train.get("grad_accum_steps", 1))
            rm_max_steps = rm_train.get("max_steps", None)
            rm_max_steps = int(rm_max_steps) if rm_max_steps is not None else None
            rm_num_workers = int(max(0, rm_train.get("num_workers", 0)))
            rm_grad_ckpt = bool(rm_train.get("gradient_checkpointing", True))

            world = accelerator.num_processes
            rank = accelerator.process_index

            if accelerator.is_main_process:
                logger.info(
                    f"[bootstrap ensemble] GPT RM parallel training: m={m}, world_size={world}, "
                    f"base_rm={base_rm}, rm_num_workers={rm_num_workers}, grad_ckpt={rm_grad_ckpt}, "
                    f"samples_per_model={rm_k}"
                )

            Ntrain = len(train_ds)

            for j in range(rank, m, world):
                out_dir = os.path.join(model_root, f"rm_{j}")
                if reuse_cache and _model_is_done(out_dir):
                    if accelerator.is_local_main_process:
                        logger.info(f"[RM train GPT] rm_{j}: cache hit at {out_dir}; skipping.")
                    continue

                idx = _sample_k_unique_indices(Ntrain, rm_k, seed=shard_seed + 10_000 * j + 17)
                if accelerator.is_local_main_process:
                    logger.info(f"[RM train GPT] rm_{j}: training on random sample size={len(idx)} out={out_dir}")

                ds_shard = train_ds.select(idx)
                train_reward_model_gpt_pairwise(
                    ds_shard,
                    base_model_name=base_rm,
                    output_dir=out_dir,
                    device=accelerator.device,
                    lr=rm_lr,
                    batch_size=rm_bs,
                    epochs=rm_epochs,
                    max_length=rm_max_len,
                    grad_accum_steps=rm_grad_accum,
                    max_steps=rm_max_steps,
                    seed=shard_seed + j,
                    num_workers=rm_num_workers,
                    gradient_checkpointing=rm_grad_ckpt,
                    logger=logger if accelerator.is_local_main_process else None,
                )
                _cleanup_cuda(aggressive=True)
                logger.info(f"[rank {rank}] finished rm_{j} at {datetime.now().isoformat()}")

            wait_for_all_reward_models(
                model_root=model_root,
                m=m,
                sleep_s=5.0,
                logger=logger if accelerator.is_main_process else None,
            )

            accelerator.wait_for_everyone()

            model_dirs = [os.path.join(model_root, f"rm_{j}") for j in range(m)]
            if accelerator.is_main_process:
                missing = [d for d in model_dirs if not _model_is_done(d)]
                if missing:
                    raise RuntimeError(f"Some GPT reward models are missing/incomplete: {missing}")
                logger.info(f"[bootstrap ensemble] all {m} GPT reward models exist under {model_root}")

            score_cfg = be_cfg.get("scoring", {})
            score_bs = int(score_cfg.get("batch_size", 32))
            score_max_len = int(score_cfg.get("max_length", 1024))
            score_num_workers = int(max(0, score_cfg.get("num_workers", 0)))

            if accelerator.is_main_process:
                logger.info(
                    f"[bootstrap ensemble] Computing consensus scores in parallel (data-sharded) (GPT RM). "
                    f"score_num_workers={score_num_workers}"
                )

            C = compute_consensus_scores_parallel_gpt(
                train_ds,
                model_dirs,
                gamma=gamma,
                device=accelerator.device,
                batch_size=score_bs,
                max_length=score_max_len,
                accelerator=accelerator,
                num_workers=score_num_workers,
                logger=logger if accelerator.is_main_process else None,
            )

            _cleanup_cuda(aggressive=True)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                sorted_indices = sorted(range(len(C)), key=lambda i: float(C[i]), reverse=True)
                with open(order_path, "w") as f:
                    json.dump(sorted_indices, f)
                logger.info(f"[bootstrap ensemble] Saved sorted indices to {order_path}")
            accelerator.wait_for_everyone()

        with open(order_path, "r") as f:
            sorted_indices = json.load(f)

        train_ds = train_ds.select(sorted_indices)
        logger.info(f"[bootstrap ensemble] Applied ordering by consensus score (desc). m={m}, gamma={gamma}, base_rm={base_rm}")

        _cleanup_cuda(aggressive=True)
        accelerator.wait_for_everyone()
    else:
        logger.info("[bootstrap ensemble] Disabled; no ordering applied.")

    train_ds_ppo = drop_to_ppo_columns(train_ds)
    eval_ds_ppo = drop_to_ppo_columns(eval_ds)

    ppo_cfg = PPOConfig(
        model_name=model_name,
        learning_rate=lr,
        log_with="wandb" if wandb_enabled else None,
        batch_size=batch_size,
        init_kl_coef=init_kl_coef,
        target_kl=target_kl,
        mini_batch_size=mini_batch_size,
        ppo_epochs=ppo_epochs,
    )

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_cfg.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(ppo_cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # ───────── NEW: enable PPO policy gradient checkpointing ─────────
    if ppo_grad_ckpt:
        enable_gradient_checkpointing_for_ppo(policy_model, logger=logger, tag="policy")
    else:
        logger.info("[ppo] gradient_checkpointing(policy) disabled by config")

    # ref model is typically no-grad; we still force cache off to avoid incompatibilities in shared codepaths
    enable_gradient_checkpointing_for_ppo(ref_model, logger=logger, tag="ref(use_cache_only)")

    ppo_trainer = PPOTrainer(
        ppo_cfg,
        policy_model,
        ref_model,
        tokenizer,
        dataset=train_ds_ppo,
        data_collator=collator,
    )

    total_batches = len(ppo_trainer.dataloader)
    eval_every = max(1, total_batches // 10)
    logger.info(f"total_batches = {total_batches}, eval_every = {eval_every}")

    val_sampler = DistributedSampler(
        eval_ds_ppo,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        eval_ds_ppo,
        sampler=val_sampler,
        batch_size=64,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
    )

    device = ppo_trainer.accelerator.device
    RM_REPO = cfg.get("reward_model", {}).get("repo", "Ray2333/gpt2-large-helpful-reward_model")

    if accelerator.is_main_process:
        _ = AutoTokenizer.from_pretrained(RM_REPO, trust_remote_code=True)
        _ = AutoModelForSequenceClassification.from_pretrained(RM_REPO, trust_remote_code=True)
    accelerator.wait_for_everyone()

    rm_tok = AutoTokenizer.from_pretrained(RM_REPO, trust_remote_code=True)
    rm_tok.pad_token_id = rm_tok.eos_token_id
    rm_model = AutoModelForSequenceClassification.from_pretrained(RM_REPO, trust_remote_code=True).eval().to(device)
    rm_model.config.pad_token_id = rm_tok.pad_token_id

    generation_kwargs = {
        "min_length": 8,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": max_response_length,
    }

    best_val_reward = float("-inf")

    for batch_idx, batch in enumerate(tqdm(ppo_trainer.dataloader), 0):
        prompts = batch["query"]
        query_tensors = batch["input_ids"]
        input_lengths = [len(ids) for ids in query_tensors]

        gen_outputs = ppo_trainer.generate(query_tensor=query_tensors, batch_size=batch_size, **generation_kwargs)

        response_tensors = [out_ids[input_len:] for out_ids, input_len in zip(gen_outputs, input_lengths)]
        generated_texts = [tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in response_tensors]
        batch["response"] = generated_texts

        full_texts = [p + " " + g for p, g in zip(prompts, generated_texts)]
        rm_inputs = rm_tok(full_texts, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)

        with bf16_autocast_if_available(device):
            logits = rm_model(**rm_inputs).logits.squeeze(-1)

        reward_tensor = logits.float().detach()
        reward_list = list(reward_tensor.unbind(dim=0))

        stats = ppo_trainer.step(query_tensors, response_tensors, reward_list)
        ppo_trainer.log_stats(stats, batch, reward_list)

        del rm_inputs, logits, reward_tensor, full_texts
        if batch_idx % 50 == 0:
            _cleanup_cuda(aggressive=False)

        if accelerator.is_main_process and wandb_enabled:
            table = wandb.Table(columns=["query", "response", "reward"])
            for q, g, r in zip(prompts, generated_texts, reward_list):
                table.add_data(q, g, float(r.item()))
            wandb.log({f"samples_step_{batch_idx}": table}, step=batch_idx)

        if (batch_idx + 1) % eval_every == 0:
            val_reward = compute_val_reward(
                ppo_trainer=ppo_trainer,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                rm_model=rm_model,
                rm_tok=rm_tok,
                accelerator=accelerator,
                generation_kwargs=generation_kwargs,
                batch_size=64,
            )

            if accelerator.is_main_process:
                if cfg.get("mlflow", {}).get("enabled", True):
                    mlflow.log_metric("val_reward", val_reward, step=batch_idx)

                if wandb_enabled:
                    wandb.log(
                        {
                            "batch_idx": batch_idx,
                            "val_reward": val_reward,
                            "hp/batch_size": batch_size,
                            "hp/learning_rate": lr,
                            "hp/init_kl_coef": init_kl_coef,
                            "hp/gradient_checkpointing": ppo_grad_ckpt,
                        },
                        step=batch_idx,
                    )

                if val_reward > best_val_reward:
                    best_val_reward = val_reward

                    ckpt_name = f"best_lr_{lr:.1e}_bs_{batch_size}_ep_{ppo_epochs}"
                    best_dir = os.path.join(output_dir, ckpt_name)
                    os.makedirs(best_dir, exist_ok=True)

                    model_to_save = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
                    model_to_save.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)

                    best_meta = {
                        "best_batch_idx": batch_idx,
                        "best_val_reward": float(val_reward),
                        "batch_size": int(batch_size),
                        "learning_rate": float(lr),
                        "init_kl_coef": float(init_kl_coef),
                        "best_dir": best_dir,
                    }
                    with open(os.path.join(best_dir, "best_metadata.json"), "w") as f:
                        json.dump(best_meta, f, indent=2)

                    logger.info(f"New best val_reward={val_reward:.4f} at step={batch_idx}, saved to {best_dir}")

    _cleanup_cuda(aggressive=True)
    logger.info("PPO (bootstrap ensemble curriculum, GPT RM) training complete")

    if accelerator.is_main_process and cfg.get("mlflow", {}).get("enabled", True):
        mlflow.end_run()
    if accelerator.is_main_process and wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
