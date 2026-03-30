import os, json, gc, math
from typing import Tuple
import yaml
import argparse
import logging
from datetime import datetime

import torch
import numpy as np
from datasets import load_dataset, load_from_disk
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


def add_prompt_len_and_sort_curriculum(train_ds, tokenizer, max_prompt_length: int, logger: logging.Logger):
    """
    Length curriculum: compute tokenized prompt length (truncated) and sort ascending.
    """
    logger.info("Applying length curriculum: computing prompt_len and sorting train set easy→hard...")

    def _len_map(ex):
        ids = tokenizer(
            ex["prompt"],
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        )["input_ids"]
        return {"prompt_len": int(len(ids))}

    # This runs on each rank (safe). Dataset.map is deterministic.
    train_ds = train_ds.map(_len_map, desc="Computing prompt_len")
    train_ds = train_ds.sort("prompt_len")

    # Keep or drop prompt_len (either is fine with remove_unused_columns=False).
    # Drop to keep dataset clean:
    try:
        train_ds = train_ds.remove_columns(["prompt_len"])
    except Exception:
        pass

    logger.info("Length curriculum applied: train set sorted by prompt length.")
    return train_ds


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

    log_dir = cfg.get("logging", {}).get("dir", "/efs/rohandeb/logs/qwen/hh-rlhf/dpo")
    logger = setup_logging(log_dir)
    logger.info("Starting DPO training (length curriculum + best-by-eval-loss checkpointing)")
    logger.info(json.dumps(cfg, indent=2, default=str))

    accelerator = Accelerator()

    # MLflow
    if cfg.get("mlflow", {}).get("enabled", True):
        os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
        os.environ.pop("MLFLOW_RUN_ID", None)

        mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "https://mlflow.rlscience.scot.amazon.dev"))
        mlflow.set_experiment(cfg.get("mlflow", {}).get("experiment", "dpo"))
        if accelerator.is_main_process:
            mlflow.start_run(
                run_name=cfg.get("mlflow", {}).get("run_name", f"dpo-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            )

    dataset_name = cfg["dataset"]["name"]
    model_name = cfg["model_name"]
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])
    num_epochs = int(cfg["training"]["num_epochs"])
    beta = float(cfg["dpo"]["beta"])
    grad_accum = int(cfg.get("training", {}).get("gradient_accumulation_steps", 1))

    num_evals = int(cfg.get("training", {}).get("num_evals", 10))
    eval_steps_cfg = cfg.get("training", {}).get("eval_steps", None)

    max_prompt_length = int(cfg["dataset"]["max_prompt_length"])
    max_response_length = int(cfg["dataset"]["max_response_length"])

    logger.info(
        f"Training config: epochs={num_epochs}, lr={lr}, batch_size={batch_size}, "
        f"grad_accum={grad_accum}, beta(fixed)={beta}"
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

    # dataset -> {prompt, chosen, rejected} (WITH DISK CACHING)
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

    # split train/val
    split_seed = int(cfg["dataset"].get("split_seed", 42))
    val_frac = float(cfg["dataset"].get("val_frac", 0.10))
    split = ds.train_test_split(test_size=val_frac, seed=split_seed)
    train_ds, val_ds = split["train"], split["test"]
    logger.info(f"Train size={len(train_ds)}  Val size={len(val_ds)}  (val_frac={val_frac})")

    # ------------------------ LENGTH CURRICULUM (easy→hard)
    train_ds = add_prompt_len_and_sort_curriculum(
        train_ds=train_ds,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        logger=logger,
    )

    # compute steps using EFFECTIVE batch size (per_device * world_size * grad_accum)
    world_size = accelerator.num_processes
    effective_bs = batch_size * world_size * grad_accum
    steps_per_epoch = max(1, int(math.ceil(len(train_ds) / effective_bs)))
    total_steps = steps_per_epoch * max(1, num_epochs)

    eval_steps = int(eval_steps_cfg) if eval_steps_cfg is not None else max(1, total_steps // max(1, num_evals))
    expected_evals = max(1, total_steps // eval_steps)

    logger.info(
        f"world_size={world_size}, effective_bs={effective_bs} "
        f"(=batch_size {batch_size} * world_size {world_size} * grad_accum {grad_accum}); "
        f"steps_per_epoch={steps_per_epoch}, total_steps≈{total_steps}, "
        f"eval_steps={eval_steps} => expected evals≈{expected_evals}"
    )

    output_dir = cfg["checkpoints"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------ BEST DIR NAME ENCODING HYPERPARAMS (PPO-style)
    best_run_dirname = (
        f"best_lr_{lr:.1e}_bs_{batch_size}_ep_{num_epochs}_beta_{beta}_ga_{grad_accum}"
        f"_mpl_{max_prompt_length}_mrl_{max_response_length}"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(cfg.get("training", {}).get("eval_batch_size", batch_size)),
        gradient_accumulation_steps=grad_accum,

        learning_rate=lr,
        num_train_epochs=num_epochs,
        logging_steps=int(cfg["training"]["logging_steps"]),
        remove_unused_columns=False,
        report_to=[],

        evaluation_strategy="steps",
        eval_steps=eval_steps,

        # disable Trainer automatic checkpoint saves
        save_strategy="no",
        dataloader_drop_last=False,
    )

    best_cb = BestValLossCheckpointCallback(
        accelerator=accelerator,
        tokenizer=tokenizer,
        output_dir=output_dir,
        best_run_dirname=best_run_dirname,
        logger=logger,
        save_metadata=True,
        hparams={
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "num_epochs": int(num_epochs),
            "beta": float(beta),
            "gradient_accumulation_steps": int(grad_accum),
            "world_size": int(world_size),
            "effective_bs": int(effective_bs),
            "max_prompt_length": int(max_prompt_length),
            "max_response_length": int(max_response_length),
        },
    )

    if accelerator.is_main_process and cfg.get("wandb", {}).get("enabled", False):
        import wandb
        wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], name=cfg["wandb"]["run_name"])

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        tokenizer=tokenizer,
        beta=beta,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[best_cb],
        max_length=int(cfg["dataset"]["max_prompt_length"] + cfg["dataset"]["max_response_length"]),
        max_prompt_length=int(cfg["dataset"]["max_prompt_length"]),
    )

    logger.info("Starting DPO training")
    trainer.train()

    # ------------------------- always save a final checkpoint for THIS hyperparameter run
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt_name = f"final_lr_{lr:.1e}_bs_{batch_size}_ep_{num_epochs}_beta_{beta}_ga_{grad_accum}"
        final_dir = os.path.join(output_dir, ckpt_name)
        os.makedirs(final_dir, exist_ok=True)

        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        final_meta = {
            "global_step": int(getattr(trainer.state, "global_step", -1)),
            "epoch": float(getattr(trainer.state, "epoch", 0.0) or 0.0),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "num_epochs": int(num_epochs),
            "beta": float(beta),
            "gradient_accumulation_steps": int(grad_accum),
            "world_size": int(accelerator.num_processes),
            "effective_bs": int(batch_size * accelerator.num_processes * grad_accum),
            "best_step": int(best_cb.best_step),
            "best_eval_loss": float(best_cb.best_eval_loss),
            "best_dir": best_cb.best_dir,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(final_dir, "final_metadata.json"), "w") as f:
            json.dump(final_meta, f, indent=2)

        logger.info(f"Final checkpoint saved to {final_dir}")
        logger.info(f"Best checkpoint (if any eval happened) is at {best_cb.best_dir}")

    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("DPO training complete (best checkpoint saved under hp-coded directory name)")

    if accelerator.is_main_process and cfg.get("mlflow", {}).get("enabled", True):
        mlflow.end_run()


if __name__ == "__main__":
    main()
