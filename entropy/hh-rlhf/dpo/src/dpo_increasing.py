#!/usr/bin/env python
"""
Curriculum-by-Entropy for DPO (bucketed, recompute after each bucket)
+ Validation-loss evaluation AFTER each bucket (distributed across all ranks)
+ Best checkpoint saved by lowest validation eval_loss (same callback as your reference code)
+ Best checkpoint directory name: best_lr_{lr}_bs_{bs}_ep_{ep}
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
# dataset helpers (from your reference code)
# ──────────────────────────────────────────────────────────────────────────────
def extract_pr(full_text: str) -> Tuple[str, str]:
    idx = full_text.rfind("Assistant:")
    if idx == -1:
        return full_text.strip(), ""
    prompt = full_text[:idx].strip()
    response = full_text[idx + len("Assistant:") :].strip()
    return prompt, response


def convert_hh(example):
    p, c = extract_pr(example["chosen"])
    _, r = extract_pr(example["rejected"])
    return {"prompt": p, "chosen": c, "rejected": r}


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
# FIXED: correct response logprob for entropy computation
# prompt/response boundary matches joint = prompt + eos + response
# and uses correct causal shift.
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
    """
    Sum log-prob of response tokens given prompt, for each example in batch.
    Uses joint = prompt + sep + response, sep = tokenizer.eos_token (if defined).

    Returns: (B,) tensor
    """
    sep = tokenizer.eos_token or ""

    joint_texts = [p + sep + r for p, r in zip(prompts, responses)]
    prefix_texts = [p + sep for p in prompts]  # includes separator, matches joint boundary

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

    # prefix length includes the inserted sep token(s)
    prefix_lens = enc_prefix["attention_mask"].sum(dim=1).to(device)  # (B,)

    outputs = model(joint_ids, attention_mask=attn_mask)
    logp = log_softmax(outputs.logits, dim=-1)  # (B, T, V)

    ll = torch.zeros(joint_ids.size(0), device=device)

    for i in range(joint_ids.size(0)):
        seq_len = int(attn_mask[i].sum().item())
        resp_start = int(prefix_lens[i].item())
        if resp_start >= seq_len:
            ll[i] = 0.0
            continue

        resp_tokens = joint_ids[i, resp_start:seq_len]                 # t = resp_start..seq_len-1
        logp_pos = logp[i, resp_start - 1: seq_len - 1, :]             # logits at t-1 predict token at t
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
    """
    Distributed entropy computation across all processes:
      p = σ(β[(Δ log π_theta) - (Δ log π_ref)]), H = -p log p -(1-p) log(1-p)
    This is inference-only: models are in eval() and this function is @torch.no_grad().
    """
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
    loader = accelerator.prepare(loader)  # shards across all processes

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

    # Gather (kept in the same style you were using; you said it runs for you)
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
    """
    Save best checkpoint according to eval_loss (lower is better).
    Saves ONLY when eval_loss improves, to output_dir/<best_run_dirname> (overwritten each time).
    """
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
        # IMPORTANT: run only on main process to avoid multi-rank clobbering
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

        # log eval_loss to MLflow (if active)
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

    log_dir = cfg.get("logging", {}).get("dir", "/efs/rohandeb/logs/entropy/qwen/hh-rlhf/dpo")
    logger = setup_logging(log_dir)
    logger.info("Starting DPO entropy curriculum training + per-bucket validation")
    logger.info(json.dumps(cfg, indent=2, default=str))

    accelerator = Accelerator()

    # MLflow (same structure as your reference code)
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

    # These were NOT hardcoded in your reference code; they were read with defaults.
    # If missing in config, defaults apply (same as your reference).
    grad_accum = int(cfg.get("training", {}).get("gradient_accumulation_steps", 1))
    eval_batch_size = int(cfg.get("training", {}).get("eval_batch_size", batch_size))

    start_buckets = int(cfg["curriculum"]["start_buckets"])
    entropy_batch = int(cfg["curriculum"]["entropy_batch_size"])

    max_prompt_length = int(cfg["dataset"]["max_prompt_length"])
    max_response_length = int(cfg["dataset"]["max_response_length"])

    # split train/val EXACTLY like your reference code (with defaults if absent)
    split_seed = int(cfg["dataset"].get("split_seed", 42))
    val_frac = float(cfg["dataset"].get("val_frac", 0.10))

    logger.info(
        f"Run cfg: buckets={start_buckets}, epochs_per_bucket={num_epochs}, lr={lr}, "
        f"batch_size={batch_size}, eval_bs={eval_batch_size}, grad_accum={grad_accum}, beta={beta}, "
        f"mpl={max_prompt_length}, mrl={max_response_length}, entropy_batch={entropy_batch}, "
        f"val_frac={val_frac}, split_seed={split_seed}"
    )

    # models
    logger.info(f"Loading models: {model_name}")
    policy_model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    device = accelerator.device
    policy_model.to(device)
    ref_model.to(device)

    policy_model.gradient_checkpointing_enable()
    logger.info("Models loaded successfully (policy gradient checkpointing enabled)")

    # dataset (WITH DISK CACHING like your reference code; dataset_dir is optional)
    dataset_dir = cfg.get("dataset", {}).get("dataset_dir", None)
    logger.info(f"Loading dataset: {dataset_name} split={cfg['dataset']['split']} dataset_dir={dataset_dir}")

    raw_ds = load_dataset_cached(
        dataset_name=dataset_name,
        split=cfg["dataset"]["split"],
        dataset_dir=dataset_dir,
        accelerator=accelerator,
        logger=logger,
    )

    if cfg["dataset"].get("max_samples"):
        raw_ds = raw_ds.select(range(cfg["dataset"]["max_samples"]))
    logger.info(f"Dataset loaded with {len(raw_ds)} samples")

    ds = raw_ds.map(convert_hh, remove_columns=raw_ds.column_names)

    split = ds.train_test_split(test_size=val_frac, seed=split_seed)
    train_pool, val_ds = split["train"], split["test"]
    logger.info(f"Train size={len(train_pool)}  Val size={len(val_ds)}  (val_frac={val_frac})")

    # output dir
    output_dir = cfg["checkpoints"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    # BEST CHECKPOINT NAME: EXACTLY lr/bs/ep only
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

        # 1) recompute entropies on remaining (distributed inference: eval + no_grad)
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

        # 2) main process selects easiest bucket, broadcast indices
        if accelerator.is_main_process:
            sorted_idx = np.argsort(ent)  # low entropy = easy
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

        # 3) Train on this bucket (disable automatic eval/checkpointing)
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
            eval_dataset=val_ds,         # needed for per-bucket trainer.evaluate()
            callbacks=[best_cb],         # saves best by eval_loss exactly like reference callback
            max_length=max_prompt_length + max_response_length,
            max_prompt_length=max_prompt_length,
        )

        logger.info(f"Stage {stage+1}: training on bucket ({len(stage_ds)} examples)")
        trainer.train()

        # 4) Evaluate ONCE after bucket training (distributed across all ranks)
        accelerator.wait_for_everyone()
        logger.info(f"Stage {stage+1}: running validation evaluate() on {len(val_ds)} examples (distributed)")
        eval_metrics = trainer.evaluate()  # triggers on_evaluate -> BestValLossCheckpointCallback
        if accelerator.is_main_process:
            logger.info(f"Stage {stage+1}: validation metrics: {json.dumps(eval_metrics, indent=2, default=str)}")

        # Optional stage-final checkpoint (NOT the best checkpoint)
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

    # Final save for this run
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
        }
        with open(os.path.join(final_dir, "final_metadata.json"), "w") as f:
            json.dump(final_meta, f, indent=2)

        logger.info(f"Final checkpoint saved to {final_dir}")
        logger.info(f"Best checkpoint is at {best_cb.best_dir}")

    if accelerator.is_main_process and cfg.get("mlflow", {}).get("enabled", True):
        mlflow.end_run()


if __name__ == "__main__":
    main()
