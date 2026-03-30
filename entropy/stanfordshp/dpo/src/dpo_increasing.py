#!/usr/bin/env python
"""
Curriculum-by-Entropy for DPO (bucketed, recompute after each bucket)
+ Validation-loss evaluation AFTER each bucket (distributed across all ranks)
+ Best checkpoint saved by lowest validation eval_loss (same callback as your reference code)
+ Best checkpoint directory name: best_lr_{lr}_bs_{bs}_ep_{ep}_bucket_{start_buckets}

THIS VERSION: Stanford SHP data handling (everything else kept the same as your chosen script).
"""

import os, json, gc, math
from typing import List, Tuple
import yaml
import argparse
import logging
from datetime import datetime

import torch
import numpy as np
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
)
from trl import DPOTrainer

import accelerate
_orig_accel_init = accelerate.Accelerator.__init__
def _patched_accel_init(self, *args, dispatch_batches=None, **kwargs):
    return _orig_accel_init(self, *args, **kwargs)
accelerate.Accelerator.__init__ = _patched_accel_init
from accelerate import Accelerator

import mlflow


# ──────────────────────────────────────────────────────────────────────────────
# logging / config
# ──────────────────────────────────────────────────────────────────────────────
def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dpo_training_log_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# dataset cache (same as before)
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
            with open(done_file, "w") as f:
                f.write("ok\n")
                f.flush()
                os.fsync(f.fileno())
            logger.info(f"Dataset saved to {dataset_dir} and _DONE written.")

    accelerator.wait_for_everyone()
    return load_from_disk(dataset_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Stanford SHP conversion (borrowed from your working SHP code)
# ──────────────────────────────────────────────────────────────────────────────
def _require_keys(example: dict, keys: list[str], ds_columns: list[str]):
    missing = [k for k in keys if k not in example]
    if missing:
        raise KeyError(f"Missing keys {missing}. Available columns: {ds_columns}")


def convert_shp(example: dict, ds_columns: list[str]) -> dict:
    """
    Convert Stanford SHP-style rows -> {"prompt","chosen","rejected"}.
    Expected/common:
      - prompt: example["history"]
      - responses: human_ref_A / human_ref_B
      - preference: score_A/score_B OR labels/label OR "A"/"B"
    """
    _require_keys(example, ["history"], ds_columns)
    prompt = str(example["history"])

    candA_keys = ["human_ref_A", "response_a", "A", "answer_a", "chosen_A", "completion_a"]
    candB_keys = ["human_ref_B", "response_b", "B", "answer_b", "chosen_B", "completion_b"]

    candA = None
    candB = None
    for k in candA_keys:
        if k in example:
            candA = example[k]
            break
    for k in candB_keys:
        if k in example:
            candB = example[k]
            break

    if candA is None or candB is None:
        if "prompt" in example and "chosen" in example and "rejected" in example:
            return {"prompt": str(example["prompt"]), "chosen": str(example["chosen"]), "rejected": str(example["rejected"])}
        raise KeyError(f"Could not find candidate response fields. Available columns: {ds_columns}")

    candA = str(candA)
    candB = str(candB)

    if "score_A" in example and "score_B" in example:
        try:
            sA = float(example["score_A"])
            sB = float(example["score_B"])
            if sA >= sB:
                return {"prompt": prompt, "chosen": candA, "rejected": candB}
            else:
                return {"prompt": prompt, "chosen": candB, "rejected": candA}
        except Exception:
            pass

    label_key = "labels" if "labels" in example else ("label" if "label" in example else None)
    if label_key is not None:
        lab = example[label_key]
        if isinstance(lab, str):
            lab_u = lab.strip().upper()
            if lab_u == "A":
                return {"prompt": prompt, "chosen": candA, "rejected": candB}
            if lab_u == "B":
                return {"prompt": prompt, "chosen": candB, "rejected": candA}
        try:
            lab_i = int(lab)
            if lab_i == 0:
                return {"prompt": prompt, "chosen": candA, "rejected": candB}
            if lab_i == 1:
                return {"prompt": prompt, "chosen": candB, "rejected": candA}
        except Exception:
            pass

    raise KeyError(
        f"No usable preference signal found (score_A/score_B or labels/label). Available columns: {ds_columns}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# FIXED: correct response logprob for entropy computation
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def response_logprob(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
    max_prompt_length: int,
    max_response_length: int,
) -> torch.Tensor:
    sep = tokenizer.eos_token or ""

    joint_texts = [p + sep + r for p, r in zip(prompts, responses)]
    prefix_texts = [p + sep for p in prompts]

    enc_joint = tokenizer(
        joint_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length + max_response_length,
        add_special_tokens=False,
    )
    enc_prefix = tokenizer(
        prefix_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    )

    joint_ids = enc_joint["input_ids"].to(device)
    attn_mask = enc_joint["attention_mask"].to(device)
    prefix_lens = enc_prefix["attention_mask"].sum(dim=1).to(device)

    outputs = model(joint_ids, attention_mask=attn_mask)
    logp = log_softmax(outputs.logits, dim=-1)

    ll = torch.zeros(joint_ids.size(0), device=device)
    for i in range(joint_ids.size(0)):
        seq_len = int(attn_mask[i].sum().item())
        resp_start = int(prefix_lens[i].item())
        if resp_start >= seq_len:
            ll[i] = 0.0
            continue
        resp_tokens = joint_ids[i, resp_start:seq_len]
        logp_pos = logp[i, resp_start - 1: seq_len - 1, :]
        ll[i] = logp_pos.gather(1, resp_tokens.unsqueeze(1)).sum()
    return ll


@torch.no_grad()
def compute_entropies(
    policy_model,
    ref_model,
    dataset: Dataset,
    tokenizer,
    beta: float,
    batch_size: int,
    device: torch.device,
    max_prompt_length: int,
    max_response_length: int,
    logger: logging.Logger,
    accelerator: Accelerator,
) -> np.ndarray | None:
    if accelerator.is_main_process:
        logger.info(f"Computing entropies for {len(dataset)} examples, batch_size={batch_size}, world_size={accelerator.num_processes}")

    policy_model.eval()
    ref_model.eval()

    idx_col = "__index_level_0__"
    if idx_col not in dataset.column_names:
        dataset = dataset.add_column(idx_col, list(range(len(dataset))))

    def collate(examples):
        return {
            "indices": [int(ex[idx_col]) for ex in examples],
            "prompt": [ex["prompt"] for ex in examples],
            "chosen": [ex["chosen"] for ex in examples],
            "rejected": [ex["rejected"] for ex in examples],
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    loader = accelerator.prepare(loader)

    local_entropies = {}

    pbar = tqdm(loader, desc="Entropy", disable=not accelerator.is_main_process)
    for batch in pbar:
        idx = batch["indices"]
        prompts = batch["prompt"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]

        log_c_theta = response_logprob(policy_model, tokenizer, prompts, chosen, device, max_prompt_length, max_response_length)
        log_r_theta = response_logprob(policy_model, tokenizer, prompts, rejected, device, max_prompt_length, max_response_length)
        log_c_ref = response_logprob(ref_model, tokenizer, prompts, chosen, device, max_prompt_length, max_response_length)
        log_r_ref = response_logprob(ref_model, tokenizer, prompts, rejected, device, max_prompt_length, max_response_length)

        z = beta * ((log_c_theta - log_r_theta) - (log_c_ref - log_r_ref))
        p = torch.sigmoid(z)
        h = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))

        h_np = h.detach().cpu().numpy()
        for i, original_idx in enumerate(idx):
            local_entropies[int(original_idx)] = float(h_np[i])

    if accelerator.num_processes > 1:
        if local_entropies:
            indices_tensor = torch.tensor(list(local_entropies.keys()), dtype=torch.long, device=device)
            values_tensor = torch.tensor(list(local_entropies.values()), dtype=torch.float32, device=device)
        else:
            indices_tensor = torch.empty(0, dtype=torch.long, device=device)
            values_tensor = torch.empty(0, dtype=torch.float32, device=device)

        all_indices = accelerator.gather(indices_tensor)
        all_values = accelerator.gather(values_tensor)

        if accelerator.is_main_process:
            entropies = np.empty(len(dataset), dtype=np.float32)
            idx_flat = all_indices.flatten().cpu().numpy()
            val_flat = all_values.flatten().cpu().numpy()
            for ii, vv in zip(idx_flat, val_flat):
                entropies[int(ii)] = float(vv)

            logger.info(
                f"Entropy done. mean={entropies.mean():.4f} std={entropies.std():.4f} "
                f"min={entropies.min():.4f} max={entropies.max():.4f}"
            )
            return entropies
        else:
            return None
    else:
        entropies = np.empty(len(dataset), dtype=np.float32)
        for ii, vv in local_entropies.items():
            entropies[int(ii)] = float(vv)
        if accelerator.is_main_process:
            logger.info(
                f"Entropy done. mean={entropies.mean():.4f} std={entropies.std():.4f} "
                f"min={entropies.min():.4f} max={entropies.max():.4f}"
            )
        return entropies


# ──────────────────────────────────────────────────────────────────────────────
# Best checkpoint callback (identical behavior to your reference code)
# ──────────────────────────────────────────────────────────────────────────────
class BestValLossCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        accelerator,
        tokenizer,
        output_dir: str,
        best_run_dirname: str,
        logger: logging.Logger = None,
        save_metadata: bool = True,
        hparams: dict | None = None,
    ):
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.best_run_dirname = best_run_dirname
        self.logger = logger
        self.save_metadata = save_metadata
        self.hparams = hparams or {}

        self.best_eval_loss = float("inf")
        self.best_step = -1

    @property
    def best_dir(self) -> str:
        return os.path.join(self.output_dir, self.best_run_dirname)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.accelerator.is_main_process:
            return

        metrics = metrics or {}
        if "eval_loss" not in metrics:
            msg = "on_evaluate: eval_loss not found in metrics; skipping best checkpoint check."
            if self.logger:
                self.logger.info(msg)
            print(msg, flush=True)
            return

        eval_loss = float(metrics["eval_loss"])
        step = int(state.global_step)

        msg = f"[EVAL] step={step} eval_loss={eval_loss:.6f} best_eval_loss={self.best_eval_loss:.6f}"
        if self.logger:
            self.logger.info(msg)
        print(msg, flush=True)

        try:
            mlflow.log_metric("eval_loss", eval_loss, step=step)
        except Exception:
            pass

        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_step = step

            best_dir = self.best_dir
            os.makedirs(best_dir, exist_ok=True)

            model = kwargs["model"]
            model_to_save = self.accelerator.unwrap_model(model)
            model_to_save.save_pretrained(best_dir)
            self.tokenizer.save_pretrained(best_dir)

            if self.save_metadata:
                meta = {
                    "best_step": self.best_step,
                    "best_eval_loss": self.best_eval_loss,
                    "timestamp": datetime.now().isoformat(),
                    "best_dir": best_dir,
                    **self.hparams,
                }
                with open(os.path.join(best_dir, "best_metadata.json"), "w") as f:
                    json.dump(meta, f, indent=2)

            msg2 = f"[BEST] new best eval_loss={eval_loss:.6f} at step={step}. Saved to {best_dir}"
            if self.logger:
                self.logger.info(msg2)
            print(msg2, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="../config/config.yaml",
        help="Path to your config YAML",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    log_dir = cfg.get("logging", {}).get("dir", "/efs/rohandeb/logs/entropy/qwen/shp/dpo")
    logger = setup_logging(log_dir)
    logger.info("Starting DPO entropy curriculum training + per-bucket validation (SHP)")
    logger.info(json.dumps(cfg, indent=2, default=str))

    accelerator = Accelerator()

    if cfg.get("mlflow", {}).get("enabled", True):
        os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
        os.environ.pop("MLFLOW_RUN_ID", None)

        mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "https://mlflow.rlscience.scot.amazon.dev"))
        mlflow.set_experiment(cfg.get("mlflow", {}).get("experiment", "dpo-curriculum"))
        if accelerator.is_main_process:
            mlflow.start_run(
                run_name=cfg.get("mlflow", {}).get("run_name", f"dpo-curriculum-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            )

    dataset_name = cfg["dataset"]["name"]
    model_name = cfg["model_name"]

    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])
    num_epochs = int(cfg["training"]["num_epochs"])
    beta = float(cfg["dpo"]["beta"])

    grad_accum = int(cfg.get("training", {}).get("gradient_accumulation_steps", 1))
    eval_batch_size = int(cfg.get("training", {}).get("eval_batch_size", batch_size))

    start_buckets = int(cfg["curriculum"]["start_buckets"])
    entropy_batch = int(cfg["curriculum"]["entropy_batch_size"])

    max_prompt_length = int(cfg["dataset"]["max_prompt_length"])
    max_response_length = int(cfg["dataset"]["max_response_length"])

    split_seed = int(cfg["dataset"].get("split_seed", 42))
    val_frac = float(cfg["dataset"].get("val_frac", 0.10))

    # SHP-specific filter (same keys/behavior as your working SHP code)
    split_name = cfg["dataset"].get("split", "train")
    input_min_text_length = int(cfg["dataset"].get("input_min_text_length", 2))
    input_max_text_length = int(cfg["dataset"].get("input_max_text_length", 512))

    logger.info(
        f"Run cfg: dataset={dataset_name}:{split_name}, buckets={start_buckets}, epochs_per_bucket={num_epochs}, lr={lr}, "
        f"batch_size={batch_size}, eval_bs={eval_batch_size}, grad_accum={grad_accum}, beta={beta}, "
        f"mpl={max_prompt_length}, mrl={max_response_length}, entropy_batch={entropy_batch}, "
        f"val_frac={val_frac}, split_seed={split_seed}, "
        f"history_len_filter=[{input_min_text_length},{input_max_text_length}]"
    )

    logger.info(f"Loading models: {model_name}")
    policy_model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # TRL's DPOTrainer inserts BOS/EOS token ids into token lists and the collator
    # expects those ids to be integers (not None). Some tokenizers ship without
    # an explicit BOS token id, so fall back safely.
    if tokenizer.bos_token is None and tokenizer.eos_token is not None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.eos_token is None and tokenizer.pad_token is not None:
        tokenizer.eos_token = tokenizer.pad_token
    if tokenizer.bos_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        tokenizer.eos_token = tokenizer.pad_token

    device = accelerator.device
    policy_model.to(device)
    ref_model.to(device)

    policy_model.gradient_checkpointing_enable()
    logger.info("Models loaded successfully (policy gradient checkpointing enabled)")

    # ──────────────────────────────────────────────────────────────────────
    # DATA LOADING (SHP) — this is the ONLY part changed vs your chosen script
    # ──────────────────────────────────────────────────────────────────────
    dataset_dir = cfg.get("dataset", {}).get("dataset_dir", None)
    logger.info(
        f"Loading dataset: {dataset_name} split={split_name} dataset_dir={dataset_dir} "
        f"(filter history length in [{input_min_text_length},{input_max_text_length}])"
    )

    raw_ds = load_dataset_cached(
        dataset_name=dataset_name,
        split=split_name,
        dataset_dir=dataset_dir,
        accelerator=accelerator,
        logger=logger,
    )

    if cfg["dataset"].get("max_samples"):
        raw_ds = raw_ds.select(range(cfg["dataset"]["max_samples"]))
    logger.info(f"Raw dataset loaded with {len(raw_ds)} samples; columns={raw_ds.column_names}")

    if "history" not in raw_ds.column_names:
        raise KeyError(f'Expected column "history". Got columns={raw_ds.column_names}')

    raw_ds = raw_ds.filter(lambda x: input_min_text_length <= len(x["history"]) <= input_max_text_length)
    logger.info(f"After history-length filter: {len(raw_ds)} samples")

    ds_columns = list(raw_ds.column_names)
    ds = raw_ds.map(
        lambda ex: convert_shp(ex, ds_columns),
        remove_columns=raw_ds.column_names,
    )
    logger.info(f"Converted to DPO schema: columns={ds.column_names}")

    split = ds.train_test_split(test_size=val_frac, seed=split_seed)
    train_pool, val_ds = split["train"], split["test"]
    logger.info(f"Train size={len(train_pool)}  Val size={len(val_ds)}  (val_frac={val_frac})")
    # ──────────────────────────────────────────────────────────────────────

    # ---------------------------------------------------------------------
    # DPOTrainer tokenization can hard-fail when the prompt boundary
    # tokenizes inconsistently due to tokenizer merge ops.
    # We remove those specific rows up-front so DPOTrainer doesn't crash.
    # ---------------------------------------------------------------------
    if bool(cfg["dataset"].get("filter_merge_op_violations", True)):
        logger.info("Filtering DPO merge-op violation examples (train/val)...")

        def _build_tokenized_answer_local(prompt: str, answer: str) -> dict:
            full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
            prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

            answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
            answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

            full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
            full_input_ids = np.array(full_tokenized["input_ids"])

            if len(full_input_ids) != len(full_concat_input_ids):
                raise ValueError("Prompt input ids and answer input ids should have same length.")

            response_token_ids_start_idx = len(prompt_input_ids)

            if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
                response_token_ids_start_idx -= 1

            prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
            prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

            answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
            answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

            return dict(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                input_ids=answer_input_ids,
                attention_mask=answer_attention_mask,
            )

        def _passes_dpo_boundary_check(feature: dict) -> bool:
            try:
                prompt = feature["prompt"]
                chosen = feature["chosen"]
                rejected = feature["rejected"]

                if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
                    return False

                chosen_tokens = _build_tokenized_answer_local(prompt, chosen)
                rejected_tokens = _build_tokenized_answer_local(prompt, rejected)

                chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
                rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])

                prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)
                chosen_prompt_slice = chosen_tokens["prompt_input_ids"][:prompt_len_input_ids]
                rejected_prompt_slice = rejected_tokens["prompt_input_ids"][:prompt_len_input_ids]

                num_diff_tokens = sum(a != b for a, b in zip(chosen_prompt_slice, rejected_prompt_slice))
                num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)

                return not (num_diff_tokens > 1 or num_diff_len > 1)
            except Exception:
                return False

        train_before = len(train_pool)
        train_pool = train_pool.filter(_passes_dpo_boundary_check, num_proc=1)
        logger.info(f"DPO boundary filter removed {train_before - len(train_pool)} rows from train.")

        val_before = len(val_ds)
        val_ds = val_ds.filter(_passes_dpo_boundary_check, num_proc=1)
        logger.info(f"DPO boundary filter removed {val_before - len(val_ds)} rows from val.")

    output_dir = cfg["checkpoints"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    # keep EXACTLY as your chosen script
    best_run_dirname = f"best_lr_{lr:.1e}_bs_{batch_size}_ep_{num_epochs}_bucket_{start_buckets}"

    world_size = accelerator.num_processes
    effective_bs = batch_size * world_size * grad_accum

    best_cb = BestValLossCheckpointCallback(
        accelerator=accelerator,
        tokenizer=tokenizer,
        output_dir=output_dir,
        best_run_dirname=best_run_dirname,
        logger=logger,
        save_metadata=True,
        hparams={
            "batch_size": int(batch_size),
            "eval_batch_size": int(eval_batch_size),
            "learning_rate": float(lr),
            "num_epochs": int(num_epochs),
            "beta": float(beta),
            "gradient_accumulation_steps": int(grad_accum),
            "world_size": int(world_size),
            "effective_bs": int(effective_bs),
            "max_prompt_length": int(max_prompt_length),
            "max_response_length": int(max_response_length),
            "start_buckets": int(start_buckets),
            "entropy_batch_size": int(entropy_batch),
            "val_frac": float(val_frac),
            "split_seed": int(split_seed),
            "input_min_text_length": int(input_min_text_length),
            "input_max_text_length": int(input_max_text_length),
            "dataset_split": str(split_name),
        },
    )

    remaining = train_pool
    metrics_file = os.path.join(log_dir, f"curriculum_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    logger.info(f"Starting curriculum loop with {len(remaining)} training examples")

    for stage in range(start_buckets):
        stages_left = start_buckets - stage
        n_remaining = len(remaining)
        if n_remaining == 0:
            break

        bucket_size = int(math.ceil(n_remaining / stages_left))
        logger.info(f"Stage {stage+1}/{start_buckets}: remaining={n_remaining} bucket_size={bucket_size}")

        ent = compute_entropies(
            policy_model=policy_model,
            ref_model=ref_model,
            dataset=remaining,
            tokenizer=tokenizer,
            beta=beta,
            batch_size=entropy_batch,
            device=device,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            logger=logger,
            accelerator=accelerator,
        )

        if accelerator.is_main_process:
            sorted_idx = np.argsort(ent)
            train_idx = sorted_idx[:bucket_size]
            keep_idx = sorted_idx[bucket_size:]

            stage_metrics = {
                "stage": int(stage + 1),
                "timestamp": datetime.now().isoformat(),
                "n_remaining_before": int(n_remaining),
                "bucket_size": int(bucket_size),
                "n_remaining_after": int(n_remaining - bucket_size),
                "entropy_stats": {
                    "mean": float(ent.mean()),
                    "std": float(ent.std()),
                    "min": float(ent.min()),
                    "max": float(ent.max()),
                    "selected_mean": float(ent[train_idx].mean()),
                    "selected_std": float(ent[train_idx].std()),
                },
            }
            with open(metrics_file, "a") as f:
                f.write(json.dumps(stage_metrics) + "\n")

            logger.info(
                f"Stage {stage+1}: entropy mean={ent.mean():.4f} selected_mean={ent[train_idx].mean():.4f} "
                f"left={n_remaining - bucket_size}"
            )
        else:
            train_idx = None
            keep_idx = None

        if accelerator.num_processes > 1:
            if accelerator.is_main_process:
                train_idx_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
                keep_idx_tensor = torch.tensor(keep_idx, dtype=torch.long, device=device)
            else:
                train_idx_tensor = torch.zeros(bucket_size, dtype=torch.long, device=device)
                keep_idx_tensor = torch.zeros(n_remaining - bucket_size, dtype=torch.long, device=device)

            torch.distributed.broadcast(train_idx_tensor, src=0)
            torch.distributed.broadcast(keep_idx_tensor, src=0)

            train_idx = train_idx_tensor.cpu().numpy()
            keep_idx = keep_idx_tensor.cpu().numpy()

        stage_ds = remaining.select(train_idx.tolist())
        remaining = remaining.select(keep_idx.tolist())

        stage_output_dir = os.path.join(output_dir, f"stage_{stage+1}")
        os.makedirs(stage_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=stage_output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum,

            learning_rate=lr,
            num_train_epochs=num_epochs,
            logging_steps=int(cfg["training"]["logging_steps"]),
            remove_unused_columns=False,
            report_to=[],

            evaluation_strategy="no",
            save_strategy="no",
            dataloader_drop_last=False,
        )

        trainer = DPOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=training_args,
            tokenizer=tokenizer,
            beta=beta,
            train_dataset=stage_ds,
            eval_dataset=val_ds,
            callbacks=[best_cb],
            max_length=max_prompt_length + max_response_length,
            max_prompt_length=max_prompt_length,
        )

        logger.info(f"Stage {stage+1}: training on bucket ({len(stage_ds)} examples)")
        trainer.train()

        accelerator.wait_for_everyone()
        logger.info(f"Stage {stage+1}: running validation evaluate() on {len(val_ds)} examples (distributed)")
        eval_metrics = trainer.evaluate()
        if accelerator.is_main_process:
            logger.info(f"Stage {stage+1}: validation metrics: {json.dumps(eval_metrics, indent=2, default=str)}")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            stage_final_dir = os.path.join(stage_output_dir, "final")
            os.makedirs(stage_final_dir, exist_ok=True)
            trainer.save_model(stage_final_dir)
            tokenizer.save_pretrained(stage_final_dir)
            logger.info(f"Stage {stage+1}: stage-final checkpoint saved to {stage_final_dir}")

        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Stage {stage+1}: done, freed trainer memory")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt_name = f"final_lr_{lr:.1e}_bs_{batch_size}_ep_{num_epochs}"
        final_dir = os.path.join(output_dir, ckpt_name)
        os.makedirs(final_dir, exist_ok=True)

        policy_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        final_meta = {
            "batch_size": int(batch_size),
            "eval_batch_size": int(eval_batch_size),
            "learning_rate": float(lr),
            "num_epochs": int(num_epochs),
            "beta": float(beta),
            "gradient_accumulation_steps": int(grad_accum),
            "world_size": int(world_size),
            "effective_bs": int(effective_bs),
            "start_buckets": int(start_buckets),
            "entropy_batch_size": int(entropy_batch),
            "val_frac": float(val_frac),
            "split_seed": int(split_seed),
            "best_step": int(best_cb.best_step),
            "best_eval_loss": float(best_cb.best_eval_loss),
            "best_dir": best_cb.best_dir,
            "timestamp": datetime.now().isoformat(),
            "dataset_split": str(split_name),
            "input_min_text_length": int(input_min_text_length),
            "input_max_text_length": int(input_max_text_length),
        }
        with open(os.path.join(final_dir, "final_metadata.json"), "w") as f:
            json.dump(final_meta, f, indent=2)

        logger.info(f"Final checkpoint saved to {final_dir}")
        logger.info(f"Best checkpoint is at {best_cb.best_dir}")

    if accelerator.is_main_process and cfg.get("mlflow", {}).get("enabled", True):
        mlflow.end_run()


if __name__ == "__main__":
    main()
