#!/usr/bin/env python
import os, json, gc, yaml, argparse, logging
from datetime import datetime
from typing import Tuple, List, Dict, Any
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

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
# safer cache locations (avoid EFS deadlocks/locks in HF cache/tokenizers)
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf/datasets")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ──────────────────────────────────────────────────────────────────────────────
# memory behavior
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
    # Avoid doing this every step; use periodically.
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
    Works under accelerate/torchrun, even if accelerate lacks gather_object.
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


def _cap_max_len(tokenizer, max_length: int) -> int:
    mm = getattr(tokenizer, "model_max_length", None)
    if mm is None:
        return int(max_length)
    try:
        mm = int(mm)
    except Exception:
        return int(max_length)
    if mm > 1_000_000:
        return int(max_length)
    return int(min(max_length, mm))


def _model_done_file(model_dir: str) -> str:
    return os.path.join(model_dir, "_DONE")


def _model_is_done(model_dir: str) -> bool:
    return os.path.isdir(model_dir) and os.path.exists(_model_done_file(model_dir))


def _sample_k_unique_indices(n: int, k: int, seed: int) -> List[int]:
    if n <= 0:
        return []
    k = int(k)
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(int(seed))
    return rng.choice(n, size=k, replace=False).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# dataset caching: load from disk if DONE exists; else download from HF and save
# ──────────────────────────────────────────────────────────────────────────────
def load_dataset_cached(
    dataset_name: str,
    split: str,
    dataset_dir: str | None,
    accelerator: Accelerator,
    logger: logging.Logger,
):
    if dataset_dir is None:
        return load_dataset(dataset_name, split=split)

    os.makedirs(dataset_dir, exist_ok=True)
    done_file = os.path.join(dataset_dir, "_DONE")

    if accelerator.is_main_process:
        if os.path.exists(done_file):
            logger.info(f"Dataset cache found: {dataset_dir} (DONE exists). Loading from disk.")
        else:
            logger.info(f"Dataset cache missing: {dataset_dir}. Rank0 downloading and saving to disk...")
            ds0 = load_dataset(dataset_name, split=split)
            ds0.save_to_disk(dataset_dir)
            _write_done(done_file)
            logger.info(f"Dataset saved to {dataset_dir} and _DONE written.")

    accelerator.wait_for_everyone()
    return load_from_disk(dataset_dir)


def build_dataset_shp(
    dataset_name: str,
    split: str,
    model_name: str,
    dataset_dir: str | None,
    accelerator: Accelerator,
    logger: logging.Logger,
    input_min_text_length: int = 2,
    input_max_text_length: int = 512,
):
    """
    Stanford SHP dataset:
      - prompt is `history`
      - responses are `human_ref_A`, `human_ref_B`
      - preference label is `labels` (1 => A preferred, 0 => B preferred)

    Returns a Dataset with:
      - "query": tokenized prompt text
      - "input_ids": tokenized prompt ids
      - "chosen"/"rejected": strings used for bootstrap ensemble reward-model training
      - keeps original SHP columns too (including 'history')
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset_cached(
        dataset_name=dataset_name,
        split=split,
        dataset_dir=dataset_dir,
        accelerator=accelerator,
        logger=logger,
    )

    ds = ds.filter(lambda x: input_min_text_length <= len(x["history"]) <= input_max_text_length)

    def _map_ex(ex):
        prompt = ex["history"]
        ids = tokenizer.encode(prompt, truncation=True, max_length=256)
        ex["input_ids"] = ids
        ex["query"] = tokenizer.decode(ids, skip_special_tokens=True)

        lab = int(ex["labels"])  # 1 => A preferred, 0 => B preferred
        a = ex["human_ref_A"]
        b = ex["human_ref_B"]
        if lab == 1:
            ex["chosen"] = a
            ex["rejected"] = b
        else:
            ex["chosen"] = b
            ex["rejected"] = a
        return ex

    ds = ds.map(_map_ex, batched=False)
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
# Bootstrap ensemble curriculum (SteamSHP T5 RM)
# ──────────────────────────────────────────────────────────────────────────────
def _shp_prompt(post: str, resp_a: str, resp_b: str) -> str:
    return (
        "POST: " + post.strip().replace("\n", " ")
        + "\n\nRESPONSE A: " + resp_a.strip().replace("\n", " ")
        + "\n\nRESPONSE B: " + resp_b.strip().replace("\n", " ")
        + "\n\nWhich response is better? RESPONSE"
    )


def _choice_token_ids(tok: T5Tokenizer):
    a_ids = tok.encode("A", add_special_tokens=False)
    b_ids = tok.encode("B", add_special_tokens=False)
    if len(a_ids) != 1 or len(b_ids) != 1:
        raise RuntimeError(f"T5 tokenizer does not encode 'A'/'B' as single tokens: A={a_ids}, B={b_ids}")
    return a_ids[0], b_ids[0]


class _PairwiseSHPDataset(Dataset):
    def __init__(self, ds):
        self._prompt = ds["history"]
        self._chosen = ds["chosen"]
        self._rejected = ds["rejected"]

    def __len__(self):
        return len(self._chosen)

    def __getitem__(self, i):
        return {"prompt": self._prompt[i], "chosen": self._chosen[i], "rejected": self._rejected[i]}


# ──────────────────────────────────────────────────────────────────────────────
# RM training collate
# - If num_workers>0: each worker inits its own tokenizer once.
# - If num_workers==0: collate uses a tokenizer passed from main process.
# ──────────────────────────────────────────────────────────────────────────────
_T5_RM_TOK = None
_T5_RM_MAXLEN = None

def _t5_rm_worker_init(worker_id: int, base_model_name: str, max_length: int):
    global _T5_RM_TOK, _T5_RM_MAXLEN
    _T5_RM_TOK = T5Tokenizer.from_pretrained(base_model_name, trust_remote_code=True, local_files_only=True)
    _T5_RM_MAXLEN = int(max_length)

def _t5_rm_collate_worker(batch: List[Dict[str, Any]]):
    global _T5_RM_TOK, _T5_RM_MAXLEN
    if _T5_RM_TOK is None or _T5_RM_MAXLEN is None:
        raise RuntimeError("T5 RM tokenizer not initialized in worker. Check worker_init_fn.")
    prompts = [b["prompt"] for b in batch]
    chosen = [b["chosen"] for b in batch]
    rejected = [b["rejected"] for b in batch]
    texts = [_shp_prompt(p, a, b) for p, a, b in zip(prompts, chosen, rejected)]
    enc = _T5_RM_TOK(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=_cap_max_len(_T5_RM_TOK, _T5_RM_MAXLEN),
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask", None)}

def _t5_rm_collate_mainproc(batch: List[Dict[str, Any]], tok: T5Tokenizer, max_length: int):
    prompts = [b["prompt"] for b in batch]
    chosen = [b["chosen"] for b in batch]
    rejected = [b["rejected"] for b in batch]
    texts = [_shp_prompt(p, a, b) for p, a, b in zip(prompts, chosen, rejected)]
    enc = tok(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=_cap_max_len(tok, int(max_length)),
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask", None)}


def train_reward_model_t5_choice(
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

    tok = T5Tokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = T5ForConditionalGeneration.from_pretrained(base_model_name).to(device)
    model.train()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    torch.manual_seed(seed)
    np.random.seed(seed)

    A_ID, _B_ID = _choice_token_ids(tok)
    decoder_start = int(getattr(model.config, "decoder_start_token_id", tok.pad_token_id))
    if decoder_start is None:
        decoder_start = int(tok.pad_token_id)

    dataset = _PairwiseSHPDataset(ds_shard)

    nw = int(max(0, num_workers))
    if nw > 0:
        collate_fn = _t5_rm_collate_worker
        worker_init_fn = partial(_t5_rm_worker_init, base_model_name=base_model_name, max_length=int(max_length))
        persistent_workers = False
        pin_memory = True
        prefetch_factor = 2
    else:
        collate_fn = partial(_t5_rm_collate_mainproc, tok=tok, max_length=int(max_length))
        worker_init_fn = None
        persistent_workers = False
        pin_memory = False
        prefetch_factor = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        drop_last=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    for ep in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"]
            if attn is not None:
                attn = attn.to(device, non_blocking=True)

            dec_in = torch.full((input_ids.size(0), 1), decoder_start, device=device, dtype=torch.long)
            targets = torch.full((input_ids.size(0),), A_ID, device=device, dtype=torch.long)

            with bf16_autocast_if_available(device):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    decoder_input_ids=dec_in,
                    use_cache=False,
                )
                logits_first = out.logits[:, 0, :]
                loss = F.cross_entropy(logits_first.float(), targets)
                loss = loss / max(1, grad_accum_steps)

            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if logger is not None and step % 50 == 0:
                logger.info(f"[RM train T5] out={output_dir} step={step} loss={float(loss.item()):.6f}")

            del input_ids, attn, dec_in, targets, out, logits_first, loss
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


# ──────────────────────────────────────────────────────────────────────────────
# Consensus scoring
# ──────────────────────────────────────────────────────────────────────────────
def consensus_collate_idx_t5(examples: List[Dict[str, Any]]):
    return {
        "idx": [int(e["__idx__"]) for e in examples],
        "prompt": [e["history"] for e in examples],
        "chosen": [e["chosen"] for e in examples],
        "rejected": [e["rejected"] for e in examples],
    }


@torch.no_grad()
def compute_consensus_scores_parallel_t5(
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
    needed = {"history", "chosen", "rejected"}
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
        collate_fn=consensus_collate_idx_t5,
        num_workers=nw,
        pin_memory=(nw > 0),
        persistent_workers=False,
        prefetch_factor=2 if nw > 0 else None,
        drop_last=False,
    )

    local_idx: List[int] = []
    local_prompt: List[str] = []
    local_chosen: List[str] = []
    local_rejected: List[str] = []

    pbar0 = tqdm(loader, desc="Consensus materialize", disable=not accelerator.is_local_main_process)
    for batch in pbar0:
        local_idx.extend(batch["idx"])
        local_prompt.extend(batch["prompt"])
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
        tok = T5Tokenizer.from_pretrained(md, trust_remote_code=True)
        model = T5ForConditionalGeneration.from_pretrained(md).eval().to(device)

        A_ID, B_ID = _choice_token_ids(tok)
        decoder_start = int(getattr(model.config, "decoder_start_token_id", tok.pad_token_id))
        if decoder_start is None:
            decoder_start = int(tok.pad_token_id)

        for start in range(0, L, batch_size):
            end = min(L, start + batch_size)
            prompts = local_prompt[start:end]
            chosen = local_chosen[start:end]
            rejected = local_rejected[start:end]
            bsz = end - start

            texts = [_shp_prompt(p, a, b) for p, a, b in zip(prompts, chosen, rejected)]
            enc = tok(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=_cap_max_len(tok, max_length),
            ).to(device)

            dec_in = torch.full((bsz, 1), decoder_start, device=device, dtype=torch.long)

            with bf16_autocast_if_available(device):
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask", None),
                    decoder_input_ids=dec_in,
                    use_cache=False,
                )
                logits_first = out.logits[:, 0, :]
                diff = (logits_first[:, A_ID] - logits_first[:, B_ID]).float().detach().cpu().numpy()

            margins[j, start:end] = diff

            del enc, dec_in, out, logits_first, diff, texts
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
    out_scores = [float("nan")] * N
    for i, c in flat:
        out_scores[i] = c
    for i in range(N):
        if not np.isfinite(out_scores[i]):
            out_scores[i] = float("-inf")

    if logger is not None:
        logger.info(f"[bootstrap ensemble] consensus scores computed for N={N} (T5)")
    return out_scores


# ──────────────────────────────────────────────────────────────────────────────
# Distributed evaluation: SteamSHP reward (T5)
# ──────────────────────────────────────────────────────────────────────────────
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

    A_ID, _B_ID = _choice_token_ids(rm_tok)

    progress_bar = tqdm(
        val_dataloader,
        desc="Validation (reward)",
        disable=not accelerator.is_local_main_process,
    )

    for batch in progress_bar:
        prompts = batch["query"]
        query_tensors = batch["input_ids"]
        input_lengths = [len(x) for x in query_tensors]

        gen_outputs = ppo_trainer.generate(
            query_tensor=query_tensors,
            batch_size=batch_size,
            **generation_kwargs,
        )

        response_tensors = [out_ids[input_len:] for out_ids, input_len in zip(gen_outputs, input_lengths)]
        generated_texts = [tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in response_tensors]

        full_texts = [_shp_prompt(p, g, ".") for p, g in zip(prompts, generated_texts)]

        rm_inputs = rm_tok(
            full_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024,
        ).to(device)

        with bf16_autocast_if_available(device):
            gen_out = rm_model.generate(
                input_ids=rm_inputs["input_ids"],
                attention_mask=rm_inputs["attention_mask"],
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            logits_A = gen_out.scores[0][:, A_ID]

        local_sum += logits_A.float().sum()
        local_count += torch.tensor(logits_A.numel(), device=device, dtype=torch.float32)

        del rm_inputs, gen_out, logits_A, full_texts, generated_texts, response_tensors, gen_outputs
        _cleanup_cuda(aggressive=False)

    accelerator.wait_for_everyone()
    global_sum = accelerator.reduce(local_sum, reduction="sum")
    global_count = accelerator.reduce(local_count, reduction="sum")

    avg_reward = (global_sum / (global_count + 1e-8)).item()
    if accelerator.is_main_process:
        print(f"[VAL] avg_reward = {avg_reward:.4f}")

    return avg_reward


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="../config/config.yaml", type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)

    log_dir = cfg.get("logging", {}).get("dir", "/efs/rohandeb/logs/ensemble/qwen/stanfordshp/ppo_ensemble")
    logger = setup_logging(log_dir)
    logger.info("Starting PPO training with bootstrap ensemble curriculum (SHP, T5 ensemble)")
    logger.info(json.dumps(cfg, indent=2, default=str))

    accelerator = Accelerator()

    dataset_name = cfg["dataset"]["name"]
    dataset_split = cfg["dataset"].get("split", "train")
    dataset_dir = cfg["dataset"].get("dataset_dir", None)

    model_name = cfg["model_name"]
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])
    ppo_epochs = int(cfg["training"]["num_epochs"])
    init_kl_coef = float(cfg["ppo"]["init_kl_coef"])
    target_kl = float(cfg["ppo"]["target_kl"])
    max_response_length = int(cfg["dataset"]["max_response_length"])
    mini_batch_size = int(cfg["ppo"]["mini_batch_size"])
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
        }
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            name=wandb_run_name,
            config=wandb_config,
        )

    raw_ds = build_dataset_shp(
        dataset_name=dataset_name,
        split=dataset_split,
        model_name=model_name,
        dataset_dir=dataset_dir,
        accelerator=accelerator,
        logger=logger,
        input_min_text_length=2,
        input_max_text_length=int(cfg["dataset"].get("max_length", 1024)),
    )
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

        base_rm = be_cfg.get("base_model_name", "stanfordnlp/SteamSHP-flan-t5-large")
        rm_k = int(be_cfg.get("rm_train_samples_per_model", 10_000))

        cache_dir = be_cfg.get("cache_dir", os.path.join(log_dir, "bootstrap_ensemble_cache"))
        reuse_cache = bool(be_cfg.get("reuse_cache", True))
        os.makedirs(cache_dir, exist_ok=True)

        order_path = os.path.join(cache_dir, f"sorted_indices_m{m}_gamma{gamma}_seed{shard_seed}.json")
        model_root = os.path.join(cache_dir, f"reward_models_m{m}_seed{shard_seed}_k{rm_k}")
        os.makedirs(model_root, exist_ok=True)

        if accelerator.is_main_process:
            _ = T5Tokenizer.from_pretrained(base_rm, trust_remote_code=True)
            _ = T5ForConditionalGeneration.from_pretrained(base_rm)
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
                    f"[bootstrap ensemble] T5 RM parallel training: m={m}, world_size={world}, "
                    f"base_rm={base_rm}, rm_num_workers={rm_num_workers}, grad_ckpt={rm_grad_ckpt}, "
                    f"samples_per_model={rm_k}"
                )
            accelerator.wait_for_everyone()

            Ntrain = len(train_ds)
            for j in range(rank, m, world):
                out_dir = os.path.join(model_root, f"rm_{j}")
                if reuse_cache and _model_is_done(out_dir):
                    if accelerator.is_local_main_process:
                        logger.info(f"[RM train T5] rm_{j}: cache hit at {out_dir}; skipping.")
                    continue

                idx = _sample_k_unique_indices(Ntrain, rm_k, seed=shard_seed + 10_000 * j + 17)
                if accelerator.is_local_main_process:
                    logger.info(f"[RM train T5] rm_{j}: training on random sample size={len(idx)} out={out_dir}")

                ds_shard = train_ds.select(idx)

                train_reward_model_t5_choice(
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

            accelerator.wait_for_everyone()

            model_dirs = [os.path.join(model_root, f"rm_{j}") for j in range(m)]
            if accelerator.is_main_process:
                missing = [d for d in model_dirs if not _model_is_done(d)]
                if missing:
                    raise RuntimeError(f"Some T5 reward models are missing/incomplete: {missing}")
                logger.info(f"[bootstrap ensemble] all {m} T5 reward models exist under {model_root}")

            score_cfg = be_cfg.get("scoring", {})
            score_bs = int(score_cfg.get("batch_size", 32))
            score_max_len = int(score_cfg.get("max_length", 1024))
            score_num_workers = int(max(0, score_cfg.get("num_workers", 0)))

            if accelerator.is_main_process:
                logger.info(
                    f"[bootstrap ensemble] Computing consensus scores in parallel (data-sharded) (T5). "
                    f"score_num_workers={score_num_workers}"
                )

            C = compute_consensus_scores_parallel_t5(
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

    ppo_trainer = PPOTrainer(
        ppo_cfg,
        policy_model,
        ref_model,
        tokenizer,
        dataset=train_ds_ppo,
        data_collator=collator,
    )

    total_batches = len(ppo_trainer.dataloader)
    num_evals = 10
    eval_every = max(1, total_batches // num_evals)
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
    RM_REPO = cfg.get("reward_model", {}).get("repo", "stanfordnlp/SteamSHP-flan-t5-large")

    if accelerator.is_main_process:
        _ = T5Tokenizer.from_pretrained(RM_REPO, trust_remote_code=True)
        _ = T5ForConditionalGeneration.from_pretrained(RM_REPO)
    accelerator.wait_for_everyone()

    rm_tok = T5Tokenizer.from_pretrained(RM_REPO, trust_remote_code=True)
    rm_tok.pad_token_id = rm_tok.eos_token_id
    rm_model = T5ForConditionalGeneration.from_pretrained(RM_REPO).eval().to(device)
    rm_model.config.pad_token_id = rm_tok.pad_token_id
    A_ID, _B_ID = _choice_token_ids(rm_tok)

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

        gen_outputs = ppo_trainer.generate(
            query_tensor=query_tensors,
            batch_size=batch_size,
            **generation_kwargs,
        )

        response_tensors = [out_ids[input_len:] for out_ids, input_len in zip(gen_outputs, input_lengths)]
        generated_texts = [tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in response_tensors]
        batch["response"] = generated_texts

        full_texts = [_shp_prompt(p, g, ".") for p, g in zip(prompts, generated_texts)]
        rm_inputs = rm_tok(
            full_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024,
        ).to(device)

        with bf16_autocast_if_available(device):
            gen_out = rm_model.generate(
                input_ids=rm_inputs["input_ids"],
                attention_mask=rm_inputs["attention_mask"],
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            batch_logits = gen_out.scores[0][:, A_ID]

        reward_list = list(batch_logits.float().detach().unbind(0))

        stats = ppo_trainer.step(query_tensors, response_tensors, reward_list)
        ppo_trainer.log_stats(stats, batch, reward_list)

        del rm_inputs, gen_out, batch_logits, full_texts
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
    logger.info("PPO (bootstrap ensemble curriculum) training complete")

    if accelerator.is_main_process and cfg.get("mlflow", {}).get("enabled", True):
        mlflow.end_run()
    if accelerator.is_main_process and wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
