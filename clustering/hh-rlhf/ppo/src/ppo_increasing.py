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
    AutoModel,
)

from torch.utils.data import DataLoader

import accelerate
from accelerate import Accelerator
import mlflow

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import wandb

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

# ──────────────────────────────────────────────────────────────────────────────
# keep the original monkey-patch (dispatch_batches arg)
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
        sample["input_ids"] = tokenizer.encode(
            extract_prompt(sample["chosen"]),
            padding=False,
            truncation=True,
            max_length=256,
        )
        sample["query"] = tokenizer.decode(
            sample["input_ids"], skip_special_tokens=True
        )
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# ──────────────────────────────────────────────────────────────────────────────
# CURRICULUM: CLUSTER-BASED SORTING (EASY → HARD)
# ──────────────────────────────────────────────────────────────────────────────
def cluster_sort_dataset(
    ds,
    *,
    n_clusters: int = 50,
    emb_batch: int = 512,
    emb_model: str = "sentence-transformers/all-mpnet-base-v2",
    emb_max_len: int = 256,
    device: str = None,
):
    """
    1) embed ds['query'] using HF AutoModel mean‐pooling
    2) cluster with MiniBatchKMeans
    3) compute TWO‐NN intrinsic dim per cluster
    4) sort ds from easiest→hardest (low→high intrinsic dim)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok_emb = AutoTokenizer.from_pretrained(emb_model)
    mdl_emb = AutoModel.from_pretrained(emb_model).to(device).eval()

    queries = ds["query"]
    all_emb = []
    for i in range(0, len(queries), emb_batch):
        batch = queries[i : i + emb_batch]
        enc = tok_emb(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=emb_max_len,
        ).to(device)
        with torch.no_grad():
            out = mdl_emb(**enc, return_dict=True).last_hidden_state  # [B, L, D]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        summed = (out * mask).sum(dim=1)  # [B, D]
        lengths = mask.sum(dim=1)  # [B, 1]
        emb = (summed / lengths).cpu().numpy()  # [B, D]
        all_emb.append(emb)
    X = np.concatenate(all_emb, axis=0)

    # clustering
    km = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=2048, init_size=3 * n_clusters
    )
    labels = km.fit_predict(X)

    # TWO‐NN intrinsic dimension per cluster
    idim = {}
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if len(idx) < 3:
            idim[k] = 1e9
            continue
        nbr = NearestNeighbors(n_neighbors=3).fit(X[idx])
        d, _ = nbr.kneighbors(X[idx])
        r1, r2 = d[:, 1], d[:, 2]
        idim[k] = 1.0 / np.log(r2 / r1 + 1e-12).mean()

    # rank clusters by idim
    order = sorted(idim, key=idim.get)
    rank = {cid: r for r, cid in enumerate(order)}
    diff = np.array([rank[c] for c in labels], dtype=int)

    # sort dataset
    sorted_idx = np.argsort(diff)
    return ds.select(sorted_idx.tolist())


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
    logger.info("Starting standard DPO (PPO) training with cluster curriculum")
    logger.info(json.dumps(cfg, indent=2, default=str))

    # outer accelerator: used for eval + process gating
    accelerator = Accelerator()

    # constants
    dataset_name = cfg["dataset"]["name"]
    model_name = cfg["model_name"]
    batch_size = cfg["training"]["batch_size"]
    lr = float(cfg["training"]["learning_rate"])
    num_epochs = cfg["training"]["num_epochs"]
    init_kl_coef = cfg["ppo"]["init_kl_coef"]
    target_kl = cfg["ppo"]["target_kl"]
    max_response_length = cfg["dataset"]["max_response_length"]
    max_prompt_length = cfg["dataset"]["max_prompt_length"]
    mini_batch_size = cfg["ppo"]["mini_batch_size"]
    output_dir = cfg["checkpoints"]["dir"]

    # cluster curriculum config
    n_clusters = cfg["cluster"]["n_clusters"]
    emb_batch = cfg["cluster"]["emb_batch"]
    emb_model = cfg["cluster"]["emb_model"]

    # build a default run name encoding key hyperparams
    default_run_name = f"dpo_cluster_bs{batch_size}_lr{lr:.1e}_kl{init_kl_coef}"
    wandb_cfg = cfg.get("wandb", {})
    wandb_run_name = wandb_cfg.get("run_name", default_run_name)

    # MLflow
    if cfg.get("mlflow", {}).get("enabled", True):
        mlflow.set_tracking_uri(
            cfg["mlflow"].get(
                "tracking_uri", "https://mlflow.rlscience.scot.amazon.dev"
            )
        )
        mlflow.set_experiment(cfg["mlflow"].get("experiment", "dpo-cluster"))
        if accelerator.is_main_process:
            mlflow.start_run(
                run_name=cfg["mlflow"].get("run_name", wandb_run_name),
            )

    # wandb
    wandb_enabled = wandb_cfg.get("enabled", True)
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

    # ──────────────────────────────────────────────────────────────────
    # Data: build, split, and CLUSTER-CURRICULUM ORDER TRAIN SET
    # ──────────────────────────────────────────────────────────────────
    raw_ds = build_dataset(dataset_name, model_name)
    print("Number of examples in raw_ds:", len(raw_ds))
    
    if cfg["dataset"].get("max_samples"):
        raw_ds = raw_ds.select(range(cfg["dataset"]["max_samples"]))

    split_seed = cfg["dataset"].get("split_seed", 42)
    # 80/20 train/val (as per comment)
    split = raw_ds.train_test_split(test_size=0.10, seed=split_seed)
    train_ds, eval_ds = split["train"], split["test"]

    # cluster curriculum ordering (easy → hard)
    train_ds = cluster_sort_dataset(
        train_ds,
        n_clusters=n_clusters,
        emb_batch=emb_batch,
        emb_model=emb_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    logger.info(
        f"Cluster curriculum applied: n_clusters={n_clusters}, emb_model={emb_model}"
    )

    # PPO config
    ppo_cfg = PPOConfig(
        model_name=model_name,
        learning_rate=lr,
        log_with="wandb" if wandb_enabled else None,
        batch_size=batch_size,
        init_kl_coef=init_kl_coef,
        target_kl=target_kl,
        mini_batch_size=mini_batch_size,
        ppo_epochs=num_epochs,
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

        # wandb samples
        if accelerator.is_main_process and wandb_enabled:
            table = wandb.Table(columns=["query", "response", "reward"])
            for q, g, r in zip(prompts, generated_texts, reward_list):
                table.add_data(q, g, float(r.item()))
            wandb.log({f"samples_step_{batch_idx}": table}, step=batch_idx)

        # periodic validation + BEST checkpointing
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
                # log current val metric
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

                # save ONLY if this is the best so far
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    best_val_step = batch_idx

                    ckpt_name = (
                        f"best_cluster_lr_{lr:.1e}_bs_{batch_size}_ep_{num_epochs}"
                    )
                    best_dir = os.path.join(output_dir, ckpt_name)
                    os.makedirs(best_dir, exist_ok=True)

                    model_to_save = ppo_trainer.accelerator.unwrap_model(
                        ppo_trainer.model
                    )
                    model_to_save.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)

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
    logger.info(
        f"DPO (PPO) training with cluster curriculum complete. "
        f"Best val_reward={best_val_reward:.4f} at step={best_val_step}"
    )

    if accelerator.is_main_process and cfg.get("mlflow", {}).get("enabled", True):
        mlflow.end_run()


if __name__ == "__main__":
    main()