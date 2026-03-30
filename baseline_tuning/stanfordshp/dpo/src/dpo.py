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


def load_dataset_cached(
    dataset_name: str,
    split: str,
    dataset_dir: str | None,
    accelerator: Accelerator,
    logger: logging.Logger,
):
    """
    PPO-style caching:
      - if dataset_dir is None: load_dataset(...) directly
      - else:
          - if dataset_dir/_DONE exists: load_from_disk(dataset_dir)
          - else: ONLY rank0 downloads + save_to_disk(dataset_dir) and writes _DONE
          - everyone waits
          - load_from_disk(dataset_dir)
    """
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


def _require_keys(example: dict, keys: list[str], ds_columns: list[str]):
    missing = [k for k in keys if k not in example]
    if missing:
        raise KeyError(
            f"Missing keys {missing}. Available columns: {ds_columns}"
        )


def convert_shp(example: dict, ds_columns: list[str]) -> dict:
    """
    Convert Stanford SHP-style rows -> {"prompt","chosen","rejected"}.

    This is the ONLY SHP-specific part.

    Expected / common schemas:
      - prompt: example["history"]
      - candidates: example["human_ref_A"], example["human_ref_B"]
      - preference signal:
          * score_A / score_B (choose higher score)
          * labels / label in {0,1} (heuristic; see below)
          * or a string like "A"/"B"

    If your dataset uses different keys, edit this function ONLY.
    """
    _require_keys(example, ["history"], ds_columns)
    prompt = str(example["history"])

    # Candidate response fields (most common)
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
        # Some preprocessed SHP variants may already be in prompt/chosen/rejected form.
        if "prompt" in example and "chosen" in example and "rejected" in example:
            return {
                "prompt": str(example["prompt"]),
                "chosen": str(example["chosen"]),
                "rejected": str(example["rejected"]),
            }
        raise KeyError(
            f"Could not find candidate response fields. Available columns: {ds_columns}"
        )

    candA = str(candA)
    candB = str(candB)

    # Prefer score-based preference if present (more reliable than label conventions)
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

    # Otherwise use labels/label if present
    label_key = "labels" if "labels" in example else ("label" if "label" in example else None)
    if label_key is not None:
        lab = example[label_key]
        # handle strings
        if isinstance(lab, str):
            lab_u = lab.strip().upper()
            if lab_u == "A":
                return {"prompt": prompt, "chosen": candA, "rejected": candB}
            if lab_u == "B":
                return {"prompt": prompt, "chosen": candB, "rejected": candA}

        # handle numeric/bool
        try:
            lab_i = int(lab)
            # Heuristic (common in pairwise datasets):
            #   0 -> A preferred, 1 -> B preferred
            if lab_i == 0:
                return {"prompt": prompt, "chosen": candA, "rejected": candB}
            if lab_i == 1:
                return {"prompt": prompt, "chosen": candB, "rejected": candA}
        except Exception:
            pass

    # If no preference signal, fail loudly (DPO requires it).
    raise KeyError(
        f"No usable preference signal found (score_A/score_B or labels/label). Available columns: {ds_columns}"
    )


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
    ):
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.best_run_dirname = best_run_dirname
        self.logger = logger
        self.save_metadata = save_metadata

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

    log_dir = cfg.get("logging", {}).get("dir", "/efs/rohandeb/logs/qwen/shp/dpo")
    logger = setup_logging(log_dir)
    logger.info("Starting DPO training (ONLY best-by-eval-loss checkpointing)")
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

    # ------------------------- DATA (SHP)
    dataset_dir = cfg.get("dataset", {}).get("dataset_dir", None)
    split_name = cfg["dataset"].get("split", "train")
    input_min_text_length = int(cfg["dataset"].get("input_min_text_length", 2))
    input_max_text_length = int(cfg["dataset"].get("input_max_text_length", 512))

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

    split_seed = int(cfg["dataset"].get("split_seed", 42))
    val_frac = float(cfg["dataset"].get("val_frac", 0.10))
    split = ds.train_test_split(test_size=val_frac, seed=split_seed)
    train_ds, val_ds = split["train"], split["test"]
    logger.info(f"Train size={len(train_ds)}  Val size={len(val_ds)}  (val_frac={val_frac})")

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

        train_before = len(train_ds)
        train_ds = train_ds.filter(_passes_dpo_boundary_check, num_proc=1)
        logger.info(f"DPO boundary filter removed {train_before - len(train_ds)} rows from train.")

        val_before = len(val_ds)
        val_ds = val_ds.filter(_passes_dpo_boundary_check, num_proc=1)
        logger.info(f"DPO boundary filter removed {val_before - len(val_ds)} rows from val.")

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

    # ONLY these three in the best directory name
    best_run_dirname = f"best_lr_{lr:.1e}_bs_{batch_size}_ep_{num_epochs}"

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

    # NO FINAL SAVING

    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("DPO training complete (only best checkpoint saved)")

    if accelerator.is_main_process and cfg.get("mlflow", {}).get("enabled", True):
        mlflow.end_run()


if __name__ == "__main__":
    main()
