import os, json, gc, yaml, argparse, logging
from datetime import datetime
from typing import Tuple
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import torch, numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from torch.utils.data import DataLoader

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
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(
    dataset_name="Anthropic/hh-rlhf",
    model_name=None,
    input_min_text_length=2,
    input_max_text_length=8,
):
    """
    Build dataset for training from Anthropic HH-RLHF.

    Returns a Dataset with:
      - "query": prompt string (conversation up to last 'Assistant:')
      - "input_ids": tokenized prompt ids (list[int] -> torch tensor via set_format)
    """
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


def collator(data):
    # list[dict] -> dict[str, list]
    return {key: [d[key] for d in data] for key in data[0]}


# ──────────────────────────────────────────────────────────────────────────────
# Distributed evaluation: compute average RM reward on eval set
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
    """
    Parallel evaluation over val_dataloader.

    Each rank:
      - iterates over its shard of val_dataloader (DistributedSampler),
      - generates responses with ppo_trainer.generate,
      - scores them with rm_model,
      - accumulates local sum and count.

    Then we all-reduce (via accelerator.reduce) to get the global average.
    """
    device = next(rm_model.parameters()).device
    rm_model.eval()

    local_sum = torch.tensor(0.0, device=device)
    local_count = torch.tensor(0.0, device=device)

    progress_bar = tqdm(
        val_dataloader,
        desc="Validation (reward)",
        disable=not accelerator.is_local_main_process,
    )

    with torch.no_grad():
        for batch in progress_bar:
            prompts = batch["query"]  # list[str]
            query_tensors = batch["input_ids"]  # list[tensor]
            input_lengths = [len(x) for x in query_tensors]

            # generate with PPO policy (TRL’s internal accelerator handles model)
            gen_outputs = ppo_trainer.generate(
                query_tensor=query_tensors,
                batch_size=batch_size,
                **generation_kwargs,
            )

            response_tensors = [
                out_ids[input_len:]
                for out_ids, input_len in zip(gen_outputs, input_lengths)
            ]

            generated_texts = [
                tokenizer.decode(r_ids, skip_special_tokens=True)
                for r_ids in response_tensors
            ]

            full_texts = [p + " " + g for p, g in zip(prompts, generated_texts)]

            rm_inputs = rm_tok(
                full_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=1024,
            ).to(device)

            logits = rm_model(**rm_inputs).logits.squeeze(-1)  # (B,)

            local_sum += logits.sum()
            local_count += torch.tensor(
                logits.numel(), device=device, dtype=torch.float32
            )

    # Reduce across all ranks
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
    parser.add_argument(
        "--config",
        "-c",
        default="../config/config.yaml",
        type=str,
        help="Path to config YAML",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    # logging
    log_dir = cfg.get("logging", {}).get(
        "dir", "/efs/rohandeb/logs/entropy/qwen/hh-rlhf/ppo"
    )
    logger = setup_logging(log_dir)
    logger.info("Starting standard PPO training")
    logger.info(json.dumps(cfg, indent=2, default=str))

    # outer accelerator: used for eval + process gating
    accelerator = Accelerator()

    # constants
    dataset_name = cfg["dataset"]["name"]
    model_name = cfg["model_name"]
    batch_size = cfg["training"]["batch_size"]
    lr = float(cfg["training"]["learning_rate"])
    ppo_epochs = cfg["training"]["num_epochs"]  # used in PPOConfig
    init_kl_coef = cfg["ppo"]["init_kl_coef"]
    target_kl = cfg["ppo"]["target_kl"]
    max_response_length = cfg["dataset"]["max_response_length"]
    max_prompt_length = cfg["dataset"]["max_prompt_length"]
    mini_batch_size = cfg["ppo"]["mini_batch_size"]
    output_dir = cfg["checkpoints"]["dir"]

    # build a default run name encoding key hyperparams
    default_run_name = f"ppo_bs{batch_size}_lr{lr:.1e}_kl{init_kl_coef}"
    wandb_cfg = cfg.get("wandb", {})
    wandb_run_name = wandb_cfg.get("run_name", default_run_name)

    # MLflow
    if cfg.get("mlflow", {}).get("enabled", True):
        mlflow.set_tracking_uri(
            cfg["mlflow"].get(
                "tracking_uri", "https://mlflow.rlscience.scot.amazon.dev"
            )
        )
        mlflow.set_experiment(cfg["mlflow"].get("experiment", "ppo-standard"))
        if accelerator.is_main_process:
            mlflow.start_run(
                run_name=cfg["mlflow"].get("run_name", wandb_run_name),
            )

    # wandb
    wandb_enabled = wandb_cfg.get("enabled", True)
    if accelerator.is_main_process and wandb_enabled:
        wandb.login(key=wandb_cfg["key"])
        # augment config with explicit hp entries
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

    # ──────────────────────────────────────────────────────────────────
    # Data: build, split, and SORT TRAIN SET BY PROMPT LENGTH (like v1)
    # ──────────────────────────────────────────────────────────────────
    raw_ds = build_dataset(dataset_name, model_name)
    print("Number of examples in raw_ds:", len(raw_ds))
    
    if cfg["dataset"].get("max_samples"):
        raw_ds = raw_ds.select(range(cfg["dataset"]["max_samples"]))

    split_seed = cfg["dataset"].get("split_seed", 42)
    split = raw_ds.train_test_split(test_size=0.10, seed=split_seed)
    train_ds, eval_ds = split["train"], split["test"]

    # sort training set by input length (ascending) for easy→hard curriculum
    lengths = [len(ids) for ids in train_ds["input_ids"]]
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    train_ds = train_ds.select(sorted_indices)

    # PPO config
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

    # models for PPO
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_cfg.model_name
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(ppo_cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # PPOTrainer (TRL handles its own accelerator)
    ppo_trainer = PPOTrainer(
        ppo_cfg,
        policy_model,
        ref_model,
        tokenizer,
        dataset=train_ds,
        data_collator=collator,
    )
    total_batches = len(ppo_trainer.dataloader)
    num_evals = 10
    eval_every = max(1, total_batches // num_evals)
    logger.info(f"total_batches = {total_batches}, eval_every = {eval_every}")

    # validation dataloader sharded with DistributedSampler using OUTER accelerator
    val_sampler = DistributedSampler(
        eval_ds,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
    )
    val_dataloader = DataLoader(
        eval_ds,
        sampler=val_sampler,
        batch_size=64,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    # reward model on same device as PPOTrainer model
    device = ppo_trainer.accelerator.device
    RM_REPO = "Ray2333/gpt2-large-helpful-reward_model"
    rm_tok = AutoTokenizer.from_pretrained(RM_REPO, trust_remote_code=True)
    rm_tok.pad_token_id = rm_tok.eos_token_id
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        RM_REPO, trust_remote_code=True
    ).eval().to(device)
    rm_model.config.pad_token_id = rm_tok.pad_token_id

    generation_kwargs = {
        "min_length": 8,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": max_response_length,
    }

    # Track best validation reward
    best_val_reward = float("-inf")
    best_val_step = -1

    # main PPO loop
    for batch_idx, batch in enumerate(tqdm(ppo_trainer.dataloader), 0):
        prompts = batch["query"]
        query_tensors = batch["input_ids"]
        input_lengths = [len(ids) for ids in query_tensors]

        # generate responses from policy
        gen_outputs = ppo_trainer.generate(
            query_tensor=query_tensors,
            batch_size=batch_size,
            **generation_kwargs,
        )

        response_tensors = [
            out_ids[input_len:]
            for out_ids, input_len in zip(gen_outputs, input_lengths)
        ]

        generated_texts = [
            tokenizer.decode(r_ids, skip_special_tokens=True)
            for r_ids in response_tensors
        ]

        batch["response"] = generated_texts

        full_texts = [p + " " + g for p, g in zip(prompts, generated_texts)]

        rm_inputs = rm_tok(
            full_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            logits = rm_model(**rm_inputs).logits  # [B, 1]

        reward_tensor = logits.squeeze(-1).detach()  # [B]
        reward_list = list(reward_tensor.unbind(dim=0))  # list[0-D tensor]

        stats = ppo_trainer.step(query_tensors, response_tensors, reward_list)
        ppo_trainer.log_stats(stats, batch, reward_list)

        # wandb table logging
        if accelerator.is_main_process and wandb_enabled:
            table = wandb.Table(columns=["query", "response", "reward"])
            for q, g, r in zip(prompts, generated_texts, reward_list):
                table.add_data(q, g, float(r.item()))
            key = f"samples_step_{batch_idx}"
            wandb.log({key: table}, step=batch_idx)

        # periodic validation + BEST checkpointing
        if (batch_idx + 1) % eval_every == 0:
            # all ranks participate for correct reduction
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
                # log current val metric
                if cfg.get("mlflow", {}).get("enabled", True):
                    mlflow.log_metric("val_reward", val_reward, step=batch_idx)

                if wandb_enabled:
                    # this gives you batch_idx vs val_reward curve,
                    # plus logs the hp values for filtering / grouping
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

                # save ONLY if this is the best so far
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    best_val_step = batch_idx

                    ckpt_name = f"best_lr_{lr:.1e}_bs_{batch_size}_ep_{ppo_epochs}"
                    best_dir = os.path.join(output_dir, ckpt_name)

                    os.makedirs(best_dir, exist_ok=True)

                    model_to_save = ppo_trainer.accelerator.unwrap_model(
                        ppo_trainer.model
                    )
                    model_to_save.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)

                    # also save hyperparameters + best val reward
                    best_meta = {
                        "best_batch_idx": batch_idx,
                        "best_val_reward": float(val_reward),
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "init_kl_coef": init_kl_coef,
                    }
                    meta_path = os.path.join(best_dir, "best_metadata.json")
                    with open(meta_path, "w") as f:
                        json.dump(best_meta, f, indent=2)

                    logger.info(
                        f"New best val_reward={val_reward:.4f} at step={batch_idx}, "
                        f"saving model and metadata to {best_dir}"
                    )

    torch.cuda.empty_cache()
    gc.collect()
    logger.info("PPO training complete")

    if accelerator.is_main_process and cfg.get("mlflow", {}).get("enabled", True):
        mlflow.end_run()


if __name__ == "__main__":
    main()
