#!/usr/bin/env python
import os, json, gc, yaml, argparse, logging
from datetime import datetime
from typing import Tuple
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import torch, numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModel

import accelerate
from accelerate import Accelerator
import mlflow

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import wandb

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


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
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(
    dataset_name: str,
    model_name: str,
    dataset_dir: str | None = None,
    input_min_text_length: int = 2,
    input_max_text_length: int = 512,
    accelerator: Accelerator | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # If not distributed / no caching dir, just load directly.
    if dataset_dir is None:
        ds = load_dataset(dataset_name, split="train")
    else:
        done_file = os.path.join(dataset_dir, "_DONE")

        # Only rank0 downloads + saves. Everyone else waits.
        if accelerator is None:
            # single process fallback
            if os.path.exists(done_file):
                ds = load_from_disk(dataset_dir)
            else:
                os.makedirs(dataset_dir, exist_ok=True)
                ds = load_dataset(dataset_name, split="train")
                ds.save_to_disk(dataset_dir)
                with open(done_file, "w") as f:
                    f.write("ok\n")
                    f.flush()
                    os.fsync(f.fileno())
        else:
            if accelerator.is_main_process:
                if not os.path.exists(done_file):
                    os.makedirs(dataset_dir, exist_ok=True)
                    ds0 = load_dataset(dataset_name, split="train")
                    ds0.save_to_disk(dataset_dir)
                    with open(done_file, "w") as f:
                        f.write("ok\n")
                        f.flush()
                        os.fsync(f.fileno())

            accelerator.wait_for_everyone()
            ds = load_from_disk(dataset_dir)

    # Filter + tokenize (runs on every rank; safe)
    ds = ds.filter(
        lambda x: input_min_text_length <= len(x["history"]) <= input_max_text_length
    )

    def tokenize(sample):
        prompt = sample["history"]
        input_ids = tokenizer.encode(prompt, truncation=True, max_length=256)
        sample["input_ids"] = input_ids
        sample["query"] = tokenizer.decode(input_ids, skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch", columns=["input_ids", "query"])
    return ds


def collator(data):
    return {k: [d[k] for d in data] for k in data[0]}


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

    sorted_idx = np.argsort(diff)
    return ds.select(sorted_idx.tolist())


# ──────────────────────────────────────────────────────────────────────────────
# Distributed evaluation: SteamSHP reward
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

    A_ID = rm_tok.encode("A", add_special_tokens=False)[0]

    progress_bar = tqdm(
        val_dataloader,
        desc="Validation (reward)",
        disable=not accelerator.is_local_main_process,
    )

    for batch in progress_bar:
        prompts = batch["query"]             # list[str]
        query_tensors = batch["input_ids"]   # list[tensor]
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

        # SteamSHP prompt template
        full_texts = [
            (
                "POST: " + p.strip().replace("\n", " ")
                + "\n\nRESPONSE A: " + g.strip().replace("\n", " ")
                + "\n\nRESPONSE B: .\n\n"
                "Which response is better? RESPONSE"
            )
            for p, g in zip(prompts, generated_texts)
        ]

        rm_inputs = rm_tok(
            full_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024,
        ).to(device)

        gen_out = rm_model.generate(
            input_ids=rm_inputs["input_ids"],
            attention_mask=rm_inputs["attention_mask"],
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

        logits_A = gen_out.scores[0][:, A_ID]  # (B,)

        local_sum += logits_A.sum()
        local_count += torch.tensor(
            logits_A.numel(), device=device, dtype=torch.float32
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
    logger.info("Starting PPO training (Stanford SHP) with cluster curriculum")
    logger.info(json.dumps(cfg, indent=2, default=str))

    # outer accelerator (for eval + process gating)
    accelerator = Accelerator()

    # constants
    dataset_name = cfg["dataset"]["name"]
    model_name = cfg["model_name"]
    batch_size = cfg["training"]["batch_size"]
    lr = float(cfg["training"]["learning_rate"])
    ppo_epochs = cfg["training"]["num_epochs"]
    init_kl_coef = cfg["ppo"]["init_kl_coef"]
    target_kl = cfg["ppo"]["target_kl"]
    max_response_length = cfg["dataset"]["max_response_length"]
    mini_batch_size = cfg["ppo"]["mini_batch_size"]
    output_dir = cfg["checkpoints"]["dir"]
    dataset_dir = cfg["dataset"].get("dataset_dir", None)

    # SHP filter bounds (same as your SHP loader)
    input_min_text_length = cfg["dataset"].get("input_min_text_length", 2)
    input_max_text_length = cfg["dataset"].get("input_max_text_length", 512)

    # cluster curriculum config (same interface as HH cluster script)
    cluster_cfg = cfg.get("cluster", {})
    n_clusters = cluster_cfg.get("n_clusters", 50)
    emb_batch = cluster_cfg.get("emb_batch", 512)
    emb_model = cluster_cfg.get("emb_model", "sentence-transformers/all-mpnet-base-v2")
    emb_max_len = cluster_cfg.get("emb_max_len", 256)

    # run name & logging configs
    default_run_name = f"ppo_cluster_bs{batch_size}_lr{lr:.1e}_kl{init_kl_coef}"
    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = wandb_cfg.get("enabled", True)
    wandb_run_name = wandb_cfg.get("run_name", default_run_name)

    # MLflow
    if cfg.get("mlflow", {}).get("enabled", True):
        mlflow.set_tracking_uri(
            cfg["mlflow"].get(
                "tracking_uri", "https://mlflow.rlscience.scot.amazon.dev"
            )
        )
        mlflow.set_experiment(cfg["mlflow"].get("experiment", "ppo-cluster"))
        if accelerator.is_main_process:
            mlflow.start_run(
                run_name=cfg["mlflow"].get("run_name", wandb_run_name),
            )

    # W&B
    if accelerator.is_main_process and wandb_enabled:
        wandb.login(key=wandb_cfg["key"])
        wandb_config = {
            **cfg,
            "training/batch_size": batch_size,
            "training/learning_rate": lr,
            "ppo/init_kl_coef": init_kl_coef,
            "cluster/n_clusters": n_clusters,
            "cluster/emb_model": emb_model,
        }
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            name=wandb_run_name,
            config=wandb_config,
        )

    # ------------------------- data
    raw_ds = build_dataset(
        dataset_name=dataset_name,
        model_name=model_name,
        dataset_dir=dataset_dir,
        input_min_text_length=input_min_text_length,
        input_max_text_length=input_max_text_length,
        accelerator=accelerator,
    )
    print("Number of examples in raw_ds:", len(raw_ds))
    if cfg["dataset"].get("max_samples"):
        raw_ds = raw_ds.select(range(cfg["dataset"]["max_samples"]))

    split_seed = cfg["dataset"].get("split_seed", 42)
    split = raw_ds.train_test_split(test_size=0.10, seed=split_seed)
    train_ds, eval_ds = split["train"], split["test"]

    # ------------------------- cluster curriculum (EASY → HARD)
    train_ds = cluster_sort_dataset(
        train_ds,
        n_clusters=n_clusters,
        emb_batch=emb_batch,
        emb_model=emb_model,
        emb_max_len=emb_max_len,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    logger.info(
        f"Cluster curriculum applied: n_clusters={n_clusters}, emb_model={emb_model}"
    )

    # ------------------------- PPO trainer
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

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_cfg.model_name
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(ppo_cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

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

    # ------------------------- validation dataloader (outer accelerator)
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

    # ------------------------- reward model (SteamSHP flan-t5-large)
    device = ppo_trainer.accelerator.device
    RM_REPO = "stanfordnlp/SteamSHP-flan-t5-large"
    rm_tok = T5Tokenizer.from_pretrained(RM_REPO, trust_remote_code=True)
    rm_tok.pad_token_id = rm_tok.eos_token_id
    rm_model = T5ForConditionalGeneration.from_pretrained(
        RM_REPO
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

    # ------------------------- training loop with best-val checkpointing
    best_val_reward = float("-inf")
    best_val_step = -1

    for batch_idx, batch in enumerate(tqdm(ppo_trainer.dataloader), 0):
        prompts = batch["query"]
        query_tensors = batch["input_ids"]
        input_lengths = [len(ids) for ids in query_tensors]

        # generate responses from PPO policy
        with torch.no_grad():
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

        # SteamSHP prompt template
        full_texts = [
            (
                "POST: " + p.strip().replace("\n", " ")
                + "\n\nRESPONSE A: " + g.strip().replace("\n", " ")
                + "\n\nRESPONSE B: .\n\n"
                "Which response is better? RESPONSE"
            )
            for p, g in zip(prompts, generated_texts)
        ]

        rm_inputs = rm_tok(
            full_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024,
        ).to(device)

        A_ID = rm_tok.encode("A", add_special_tokens=False)[0]

        with torch.no_grad():
            gen_out = rm_model.generate(
                input_ids=rm_inputs["input_ids"],
                attention_mask=rm_inputs["attention_mask"],
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )

        batch_logits = gen_out.scores[0][:, A_ID]  # [B]
        reward_list = list(batch_logits.detach().unbind(0))

        stats = ppo_trainer.step(query_tensors, response_tensors, reward_list)
        ppo_trainer.log_stats(stats, batch, reward_list)

        # W&B sample table
        if accelerator.is_main_process and wandb_enabled:
            table = wandb.Table(columns=["query", "response", "reward"])
            for q, g, r in zip(prompts, generated_texts, reward_list):
                table.add_data(q, g, float(r.item()))
            key = f"samples_step_{batch_idx}"
            wandb.log({key: table}, step=batch_idx)

        # periodic validation + best checkpoint
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
                    best_val_step = batch_idx

                    ckpt_name = f"best_cluster_lr_{lr:.1e}_bs_{batch_size}_ep_{ppo_epochs}"
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
                        "cluster": {
                            "n_clusters": n_clusters,
                            "emb_batch": emb_batch,
                            "emb_model": emb_model,
                            "emb_max_len": emb_max_len,
                        },
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
    if accelerator.is_main_process and wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
