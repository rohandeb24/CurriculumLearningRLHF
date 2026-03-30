#!/usr/bin/env python
import os, json, gc, yaml, argparse, logging, math, random
from datetime import datetime
from typing import Tuple, List, Dict, Literal, Any, Optional

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import accelerate
from accelerate import Accelerator
import mlflow
import wandb

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead


# ──────────────────────────────────────────────────────────────────────────────
# monkey-patch Accelerator.__init__ to keep dispatch_batches arg
# ──────────────────────────────────────────────────────────────────────────────
_orig_accel_init = accelerate.Accelerator.__init__
def _patched_accel_init(self, *args, dispatch_batches=None, **kwargs):
    return _orig_accel_init(self, *args, **kwargs)
accelerate.Accelerator.__init__ = _patched_accel_init


# keep memory behavior sane on large batches
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128",
)


# ──────────────────────────────────────────────────────────────────────────────
# utilities
# ──────────────────────────────────────────────────────────────────────────────
def setup_logging(log_dir: str):
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(
    dataset_name: str,
    model_name: str,
    dataset_dir: Optional[str] = None,
    accelerator: Optional[Accelerator] = None,
    max_prompt_length: int = 256,
):
    """
    hh-rlhf -> build columns:
      - query: prompt string (conversation up to last 'Assistant:')
      - input_ids: tokenized prompt ids
    Adds rank0 caching/barrier behavior like code #2 when dataset_dir is provided.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def extract_prompt(conv: str) -> str:
        idx = conv.rfind("Assistant:")
        return conv[:idx].strip() if idx != -1 else conv.strip()

    # ------------------- load (optionally cached like code #2)
    if dataset_dir is None:
        ds = load_dataset(dataset_name, split="train")
    else:
        done_file = os.path.join(dataset_dir, "_DONE")
        if accelerator is None:
            if os.path.exists(done_file):
                ds = load_from_disk(dataset_dir)
            else:
                os.makedirs(dataset_dir, exist_ok=True)
                ds0 = load_dataset(dataset_name, split="train")
                ds0.save_to_disk(dataset_dir)
                with open(done_file, "w") as f:
                    f.write("ok\n")
                    f.flush()
                    os.fsync(f.fileno())
                ds = load_from_disk(dataset_dir)
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

    # ------------------- tokenize (runs on every rank; safe)
    def tokenize(sample):
        ids = tokenizer.encode(
            extract_prompt(sample["chosen"]),
            padding=False,
            truncation=True,
            max_length=max_prompt_length,
        )
        sample["input_ids"] = ids
        sample["query"] = tokenizer.decode(ids, skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch", columns=["input_ids", "query"])
    return ds


def collator(data):
    # list[dict] -> dict[str, list]
    return {key: [d[key] for d in data] for key in data[0]}


# PEC collate must be top-level picklable
def pec_collate(examples: List[Dict[str, Any]]):
    return {
        "idx": [int(ex["__pec_idx__"]) for ex in examples],
        "query": [ex["query"] for ex in examples],
    }


def _unwrap_module(m):
    return m.module if hasattr(m, "module") else m


def _get_base_model(m):
    m0 = _unwrap_module(m)
    return getattr(m0, "pretrained_model", m0)


def _get_v_head(m):
    m0 = _unwrap_module(m)
    if hasattr(m0, "v_head"):
        return m0.v_head
    raise RuntimeError("Model has no v_head; expected AutoModelForCausalLMWithValueHead.")


# ──────────────────────────────────────────────────────────────────────────────
# Distributed evaluation: compute average RM reward on eval set (RUN ONCE PER STAGE)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_val_reward(
    ppo_trainer,
    val_dataloader,
    tokenizer,
    rm_model,
    rm_tok,
    accelerator: Accelerator,
    generation_kwargs,
    batch_size: int = 16,
):
    device = next(rm_model.parameters()).device
    rm_model.eval()

    local_sum = torch.tensor(0.0, device=device)
    local_count = torch.tensor(0.0, device=device)

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
        local_count += torch.tensor(logits.numel(), device=device, dtype=torch.float32)

    accelerator.wait_for_everyone()
    global_sum = accelerator.reduce(local_sum, reduction="sum")
    global_count = accelerator.reduce(local_count, reduction="sum")

    avg_reward = (global_sum / (global_count + 1e-8)).item()
    if accelerator.is_main_process:
        print(f"[VAL] avg_reward = {avg_reward:.4f}")

    return avg_reward


# ──────────────────────────────────────────────────────────────────────────────
# PEC scoring helpers
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _rollout_once_generate_full(
    model,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int,
    max_prompt_length: int = 256,
) -> torch.Tensor:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    ).to(device)

    gen_model = _get_base_model(model)
    out = gen_model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return out


@torch.no_grad()
def _logprobs_values_for_generated(
    policy_model,
    ref_model,
    tokenizer: AutoTokenizer,
    gen_ids: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    input_ids = gen_ids.to(device)
    attention_mask = (gen_ids != tokenizer.pad_token_id).to(device).long()

    pol_base = _get_base_model(policy_model)
    ref_base = _get_base_model(ref_model)

    pol_out = pol_base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        output_hidden_states=True,
    )
    logits_pol = pol_out.logits
    last_h_pol = pol_out.hidden_states[-1]

    v_head = _get_v_head(policy_model)
    values = v_head(last_h_pol).squeeze(-1)  # [B, T]

    ref_out = ref_base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits_ref = ref_out.logits

    tgt = input_ids[:, 1:]                # [B, T-1]
    mask_shift = attention_mask[:, 1:]    # [B, T-1]

    logp_pol = F.log_softmax(logits_pol[:, :-1, :], dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    logp_ref = F.log_softmax(logits_ref[:, :-1, :], dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    return {
        "logp_pol": logp_pol * mask_shift,
        "logp_ref": logp_ref * mask_shift,
        "values": values,
        "mask_shift": mask_shift,
    }


@torch.no_grad()
def _terminal_external_rewards(
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    text_tokenizer: AutoTokenizer,
    full_gen_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    texts = text_tokenizer.batch_decode(full_gen_ids, skip_special_tokens=True)
    rm_inputs = reward_tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    ).to(device)
    logits = reward_model(**rm_inputs).logits.squeeze(-1)  # [B]
    return logits


@torch.no_grad()
def compute_pec_scores_parallel_fixed_gather(
    *,
    policy_model,
    ref_model,
    tokenizer: AutoTokenizer,
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    dataset: Dataset,
    accelerator: Accelerator,
    device: torch.device,
    beta: float,
    gamma: float,
    lam: float,
    max_new_tokens: int,
    max_prompt_length: int,
    pec_batch_size: int,
    score_type: Literal["TD", "ADV"] = "TD",
    logger=None,
) -> np.ndarray | None:
    _unwrap_module(policy_model).eval()
    _unwrap_module(ref_model).eval()
    reward_model.eval()

    indexed = dataset.add_column("__pec_idx__", list(range(len(dataset))))

    sampler = DistributedSampler(
        indexed,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
    )

    loader = DataLoader(
        indexed,
        sampler=sampler,
        batch_size=pec_batch_size,
        collate_fn=pec_collate,
        num_workers=2,
        pin_memory=True,
    )

    if logger and accelerator.is_main_process:
        logger.info(
            f"[PEC] scoring n={len(indexed)} score_type={score_type} pec_batch_size={pec_batch_size} "
            f"num_ranks={accelerator.num_processes} (fixed-size gather)"
        )

    num_local = getattr(sampler, "num_samples", len(sampler))
    local_idx = torch.full((num_local,), -1, dtype=torch.long, device=device)
    local_sc = torch.full((num_local,), 0.0, dtype=torch.float32, device=device)
    write_pos = 0

    pbar = tqdm(loader, desc="PEC scoring", disable=not accelerator.is_local_main_process)

    for batch in pbar:
        idxs: List[int] = batch["idx"]
        prompts: List[str] = batch["query"]

        gen_ids = _rollout_once_generate_full(
            model=policy_model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
        )

        stats = _logprobs_values_for_generated(
            policy_model=policy_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            gen_ids=gen_ids,
            device=device,
        )
        logp_pol = stats["logp_pol"]
        logp_ref = stats["logp_ref"]
        values = stats["values"]
        mask_shift = stats["mask_shift"]

        B = gen_ids.size(0)
        Tm1 = logp_pol.size(1)

        enc_prompts = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        ).to(device)
        prompt_lens = enc_prompts["attention_mask"].sum(dim=1)  # [B]

        arange = torch.arange(Tm1, device=device).unsqueeze(0).expand(B, Tm1)
        resp_start = (prompt_lens - 1).unsqueeze(1)
        mask_resp = (arange >= resp_start) & (mask_shift > 0)

        m_t = mask_resp.float()
        kl_t = (logp_pol - logp_ref) * m_t

        r_ext = _terminal_external_rewards(
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            text_tokenizer=tokenizer,
            full_gen_ids=gen_ids,
            device=device,
        )

        lengths = (gen_ids != tokenizer.pad_token_id).sum(dim=1)  # [B]
        last_k = (lengths - 2).clamp(min=0)
        term_mask = torch.zeros_like(m_t)
        term_mask.scatter_(1, last_k.unsqueeze(1), 1.0)

        r_tot = term_mask * r_ext.unsqueeze(1) - beta * kl_t

        V_t, V_tp1 = values[:, :-1], values[:, 1:]
        m_tp1 = torch.zeros_like(m_t)
        m_tp1[:, :-1] = m_t[:, 1:]

        delta = (r_tot + gamma * m_tp1 * V_tp1 - V_t) * m_t

        adv = torch.zeros_like(delta)
        for i in range(B):
            running = 0.0
            for k in range(Tm1 - 1, -1, -1):
                if m_t[i, k] > 0:
                    running = delta[i, k] + gamma * lam * m_tp1[i, k] * running
                    adv[i, k] = running
                else:
                    running = 0.0

        if score_type.upper() == "TD":
            abs_tok = delta.abs()
        elif score_type.upper() == "ADV":
            abs_tok = adv.abs()
        else:
            raise ValueError(f"Unknown score_type={score_type}. Use 'TD' or 'ADV'.")

        denom = m_t.sum(dim=1).clamp(min=1.0)
        sent_scores = (abs_tok.sum(dim=1) / denom).detach().float()  # [B]

        for j, s in zip(idxs, sent_scores):
            if write_pos < num_local:
                local_idx[write_pos] = int(j)
                local_sc[write_pos] = float(s.item())
                write_pos += 1

        del gen_ids, stats, logp_pol, logp_ref, values, mask_shift
        del enc_prompts, prompt_lens, arange, resp_start, mask_resp, m_t, kl_t
        del r_ext, lengths, last_k, term_mask, r_tot, V_t, V_tp1, m_tp1
        del delta, adv, abs_tok, denom, sent_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    all_idx = accelerator.gather(local_idx)
    all_sc = accelerator.gather(local_sc)

    if not accelerator.is_main_process:
        return None

    N = len(dataset)
    scores = np.empty(N, dtype=np.float32)
    filled = np.zeros(N, dtype=np.bool_)

    all_idx_cpu = all_idx.detach().cpu().numpy()
    all_sc_cpu = all_sc.detach().cpu().numpy()

    for j, s in zip(all_idx_cpu, all_sc_cpu):
        j = int(j)
        if 0 <= j < N and not filled[j]:
            scores[j] = float(s)
            filled[j] = True

    if not filled.all():
        scores[~filled] = np.float32(np.inf)
        if logger:
            logger.warning(f"[PEC] Missing scores for {int((~filled).sum())} items; set to +inf.")

    return scores


def _broadcast_indices(
    train_idx: np.ndarray,
    keep_idx: np.ndarray,
    device: torch.device,
    accelerator: Accelerator,
) -> Tuple[np.ndarray, np.ndarray]:
    if accelerator.num_processes <= 1:
        return train_idx, keep_idx

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return train_idx, keep_idx

    if accelerator.is_main_process:
        t_train = torch.tensor(train_idx, dtype=torch.long, device=device)
        t_keep = torch.tensor(keep_idx, dtype=torch.long, device=device)
    else:
        t_train = torch.empty(len(train_idx), dtype=torch.long, device=device)
        t_keep = torch.empty(len(keep_idx), dtype=torch.long, device=device)

    torch.distributed.broadcast(t_train, src=0)
    torch.distributed.broadcast(t_keep, src=0)

    return t_train.cpu().numpy(), t_keep.cpu().numpy()


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

    log_dir = cfg.get("logging", {}).get("dir", "/efs/rohandeb/logs/pec/qwen/hh-rlhf/ppo")
    logger = setup_logging(log_dir)
    logger.info("Starting PPO training with Dynamic PEC curriculum (trainer init once)")
    logger.info(json.dumps(cfg, indent=2, default=str))

    # Use an outer accelerator ONLY for dataset caching + early rank gating (like code #2).
    # After PPOTrainer is created, we use ppo_trainer.accelerator for all distributed ops.
    cache_accelerator = Accelerator()

    seed = int(cfg.get("seed", cfg.get("dataset", {}).get("split_seed", 42)))
    seed_everything(seed)

    dataset_name = cfg["dataset"]["name"]
    model_name = cfg["model_name"]
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])
    ppo_epochs = int(cfg["training"]["num_epochs"])
    init_kl_coef = float(cfg["ppo"]["init_kl_coef"])
    target_kl = float(cfg["ppo"]["target_kl"])
    max_response_length = int(cfg["dataset"]["max_response_length"])
    max_prompt_length = int(cfg["dataset"]["max_prompt_length"])
    mini_batch_size = int(cfg["ppo"]["mini_batch_size"])
    output_dir = cfg["checkpoints"]["dir"]

    # optional dataset caching directory (same idea as code #2)
    dataset_dir = cfg.get("dataset", {}).get("dataset_dir", None)

    # curriculum knobs
    cur_cfg = cfg.get("curriculum", {})
    start_buckets = int(cur_cfg.get("start_buckets", 10))
    pec_batch_size = int(cur_cfg.get("pec_batch_size", 64))
    pec_score_type = str(cur_cfg.get("pec_score_type", "TD")).upper()
    gamma = float(cur_cfg.get("gamma", 1.0))
    lam = float(cur_cfg.get("gae_lambda", 0.95))

    default_run_name = f"ppoPEC_bs{batch_size}_lr{lr:.1e}_kl{init_kl_coef}_b{start_buckets}_{pec_score_type}"
    wandb_cfg = cfg.get("wandb", {})
    wandb_run_name = wandb_cfg.get("run_name", default_run_name)

    if cfg.get("mlflow", {}).get("enabled", True):
        mlflow.set_tracking_uri(cfg["mlflow"].get("tracking_uri", "https://mlflow.rlscience.scot.amazon.dev"))
        mlflow.set_experiment(cfg["mlflow"].get("experiment", "ppo-pec"))
        if cache_accelerator.is_main_process:
            mlflow.start_run(run_name=cfg["mlflow"].get("run_name", wandb_run_name))

    wandb_enabled = bool(wandb_cfg.get("enabled", True))
    if cache_accelerator.is_main_process and wandb_enabled:
        wandb.login(key=wandb_cfg["key"])
        wandb_config = {
            **cfg,
            "training/batch_size": batch_size,
            "training/learning_rate": lr,
            "ppo/init_kl_coef": init_kl_coef,
            "curriculum/start_buckets": start_buckets,
            "curriculum/pec_batch_size": pec_batch_size,
            "curriculum/pec_score_type": pec_score_type,
            "curriculum/gamma": gamma,
            "curriculum/gae_lambda": lam,
        }
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            name=wandb_run_name,
            config=wandb_config,
        )

    # ------------------------- data (rank0 cache + barrier like code #2)
    raw_ds = build_dataset(
        dataset_name=dataset_name,
        model_name=model_name,
        dataset_dir=dataset_dir,
        accelerator=cache_accelerator,
        max_prompt_length=max_prompt_length,
    )
    logger.info(f"Number of examples in raw_ds: {len(raw_ds)}")

    if cfg.get("dataset", {}).get("max_samples"):
        raw_ds = raw_ds.select(range(int(cfg["dataset"]["max_samples"])))

    split_seed = int(cfg.get("dataset", {}).get("split_seed", 42))
    split = raw_ds.train_test_split(test_size=0.10, seed=split_seed)
    train_ds, eval_ds = split["train"], split["test"]

    # ------------------------- PPO config + trainer
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

    tokenizer = AutoTokenizer.from_pretrained(ppo_cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_cfg.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_cfg.model_name)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    ppo_trainer = PPOTrainer(
        ppo_cfg,
        policy_model,
        ref_model,
        tokenizer,
        dataset=train_ds,   # only for init
        data_collator=collator,
    )

    # From here on, use the trainer accelerator everywhere (no mixing contexts).
    accelerator = ppo_trainer.accelerator
    device = accelerator.device

    # ------------------------- reward model (unchanged from your code #1)
    RM_REPO = cfg.get("reward_model", {}).get("repo", "Ray2333/gpt2-large-helpful-reward_model")
    rm_tok = AutoTokenizer.from_pretrained(RM_REPO, trust_remote_code=True)
    if rm_tok.pad_token_id is None:
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

    # ------------------------- validation dataloader (use trainer accelerator ranks)
    val_sampler = DistributedSampler(
        eval_ds,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        eval_ds,
        sampler=val_sampler,
        batch_size=64,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    best_val_reward = float("-inf")
    best_val_stage = -1

    remaining = train_ds
    logger.info(
        f"PEC curriculum: start_buckets={start_buckets}, score_type={pec_score_type}, "
        f"pec_batch_size={pec_batch_size}, total_train={len(remaining)}"
    )

    global_step = 0

    for stage in range(start_buckets):
        stages_left = start_buckets - stage
        n_remaining = len(remaining)
        if n_remaining == 0:
            break

        bucket_size = int(math.ceil(n_remaining / stages_left))
        logger.info(f"[Stage {stage+1}/{start_buckets}] remaining={n_remaining}, bucket_size={bucket_size}")

        # ---------------- PEC scoring (trainer accelerator, fixed gather)
        scores = compute_pec_scores_parallel_fixed_gather(
            policy_model=ppo_trainer.model,
            ref_model=getattr(ppo_trainer, "ref_model", ref_model),
            tokenizer=tokenizer,
            reward_model=rm_model,
            reward_tokenizer=rm_tok,
            dataset=remaining,
            accelerator=accelerator,
            device=device,
            beta=init_kl_coef,
            gamma=gamma,
            lam=lam,
            max_new_tokens=max_response_length,
            max_prompt_length=max_prompt_length,
            pec_batch_size=pec_batch_size,
            score_type=pec_score_type,
            logger=logger,
        )

        if accelerator.is_main_process:
            sorted_idx = np.argsort(scores)
            train_idx = sorted_idx[:bucket_size]
            keep_idx = sorted_idx[bucket_size:]
        else:
            train_idx = np.empty(bucket_size, dtype=np.int64)
            keep_idx = np.empty(n_remaining - bucket_size, dtype=np.int64)

        train_idx, keep_idx = _broadcast_indices(train_idx, keep_idx, device=device, accelerator=accelerator)

        stage_ds = remaining.select(train_idx.tolist())
        remaining = remaining.select(keep_idx.tolist())

        # length sort inside stage (keep your behavior)
        lengths = [len(ids) for ids in stage_ds["input_ids"]]
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        stage_ds = stage_ds.select(sorted_indices)

        # ---------------- stage dataloader FIX:
        # DistributedSampler + drop_last=True to guarantee each rank has the same number of batches,
        # and DO NOT call accelerator.prepare() (avoid accidental double-sharding).
        stage_sampler = DistributedSampler(
            stage_ds,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
            drop_last=True,
        )
        stage_loader = DataLoader(
            stage_ds,
            sampler=stage_sampler,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        logger.info(
            f"[Stage {stage+1}] stage_ds={len(stage_ds)} "
            f"sampler_num_samples={getattr(stage_sampler, 'num_samples', 'NA')} "
            f"batches_per_rank={len(stage_loader)} (drop_last=True)"
        )

        for batch in tqdm(
            stage_loader,
            desc=f"Stage {stage+1} PPO",
            disable=not accelerator.is_local_main_process,
        ):
            prompts = batch["query"]
            query_tensors = batch["input_ids"]
            input_lengths = [len(ids) for ids in query_tensors]

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

            full_texts = [p + " " + g for p, g in zip(prompts, generated_texts)]
            rm_inputs = rm_tok(
                full_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=1024,
            ).to(device)

            with torch.no_grad():
                logits = rm_model(**rm_inputs).logits.squeeze(-1)

            reward_list = list(logits.detach().unbind(dim=0))

            stats = ppo_trainer.step(query_tensors, response_tensors, reward_list)
            ppo_trainer.log_stats(stats, batch, reward_list)

            if accelerator.is_main_process and wandb_enabled:
                table = wandb.Table(columns=["stage", "global_step", "query", "response", "reward"])
                for q, g, r in zip(prompts, generated_texts, reward_list):
                    table.add_data(stage + 1, global_step, q, g, float(r.item()))
                wandb.log({f"samples/stage_{stage+1}": table}, step=global_step)

            global_step += 1

        # ---------------- eval once per stage (all ranks participate; trainer accelerator reduces)
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
                mlflow.log_metric("val_reward", val_reward, step=stage)

            if wandb_enabled:
                wandb.log(
                    {
                        "stage": stage + 1,
                        "val_reward": val_reward,
                        "hp/batch_size": batch_size,
                        "hp/learning_rate": lr,
                        "hp/init_kl_coef": init_kl_coef,
                        "curriculum/score_type": pec_score_type,
                        "curriculum/pec_batch_size": pec_batch_size,
                    },
                    step=stage,
                )

            if val_reward > best_val_reward:
                best_val_reward = float(val_reward)
                best_val_stage = int(stage)

                ckpt_name = f"best_lr_{lr:.1e}_bs_{batch_size}_ep_{ppo_epochs}_bk_{start_buckets}"
                best_dir = os.path.join(output_dir, ckpt_name)
                os.makedirs(best_dir, exist_ok=True)

                model_to_save = accelerator.unwrap_model(ppo_trainer.model)
                model_to_save.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

                best_meta = {
                    "best_stage": best_val_stage + 1,
                    "best_val_reward": best_val_reward,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "init_kl_coef": init_kl_coef,
                    "pec_score_type": pec_score_type,
                    "pec_batch_size": pec_batch_size,
                    "gamma": gamma,
                    "gae_lambda": lam,
                }
                with open(os.path.join(best_dir, "best_metadata.json"), "w") as f:
                    json.dump(best_meta, f, indent=2)

                logger.info(
                    f"New best val_reward={best_val_reward:.4f} at stage={best_val_stage+1}. Saved to {best_dir}"
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"[Stage {stage+1}] complete; remaining={len(remaining)}")

    if accelerator.is_main_process:
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        model_to_save = accelerator.unwrap_model(ppo_trainer.model)
        model_to_save.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Final checkpoint saved to {final_dir}")
        logger.info(f"Best val_reward={best_val_reward:.4f} at stage={best_val_stage+1}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("PPO + PEC training complete")

    if accelerator.is_main_process and cfg.get("mlflow", {}).get("enabled", True):
        mlflow.end_run()
    if accelerator.is_main_process and wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
