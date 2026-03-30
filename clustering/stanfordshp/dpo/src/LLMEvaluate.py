#!/usr/bin/env python3
import os, json, gc, yaml, argparse, logging
from datetime import datetime
from textwrap import wrap

import torch
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import accelerate
from accelerate import Accelerator

from openai import OpenAI


# ──────────────────────────────────────────────────────────────────────────────
# Bedrock OpenAI-compatible client (Claude on Bedrock endpoint)
# ──────────────────────────────────────────────────────────────────────────────
CLAUDE_MODEL_ID: str = ""
OPENAI_ENDPOINT: str = ""
API_KEY = ""

openai_client = OpenAI(base_url=OPENAI_ENDPOINT, api_key=API_KEY)


# ──────────────────────────────────────────────────────────────────────────────
# accelerate monkey patch (keeps dispatch_batches arg)
# ──────────────────────────────────────────────────────────────────────────────
_orig_accel_init = accelerate.Accelerator.__init__
def _patched_accel_init(self, *args, dispatch_batches=None, **kwargs):
    return _orig_accel_init(self, *args, **kwargs)
accelerate.Accelerator.__init__ = _patched_accel_init


# ──────────────────────────────────────────────────────────────────────────────
# utilities
# ──────────────────────────────────────────────────────────────────────────────
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(log_dir: str, rank: int) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dpo_eval_{ts}_rank{rank}.log")

    # Avoid duplicated handlers on reruns
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(f"rank{rank}")
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def evaluate_pair(context: str, a: str, b: str) -> str:
    prompt = f"""
You are given a conversation context and two possible assistant continuations.
Choose which continuation is better according to the stanfordnlp/SHP preference dataset.
Respond with **A** or **B** on the first line, then a one-sentence justification.

Context:
{context}

Continuation A:
{a}

Continuation B:
{b}

Answer:
""".strip()

    resp = openai_client.chat.completions.create(
        model=CLAUDE_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=1.0,
    )
    return resp.choices[0].message.content

def parse_first_bold_choice(verdict_text: str) -> str:
    """
    Count using ONLY the first **...** token on the first non-empty line.
    Returns: "A", "B", "NEITHER", or "UNKNOWN".
    """
    if not verdict_text or not verdict_text.strip():
        return "UNKNOWN"

    lines = [ln.strip() for ln in verdict_text.splitlines() if ln.strip()]
    first_line = lines[0] if lines else ""

    i = first_line.find("**")
    if i != -1:
        j = first_line.find("**", i + 2)
        if j != -1:
            token = first_line[i + 2 : j].strip()
        else:
            token = first_line[i + 2 :].strip()
    else:
        token = first_line.split()[0] if first_line else ""

    t = token.strip().upper()
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    if t.startswith("NEITHER") or t.startswith("TIE") or t.startswith("DRAW"):
        return "NEITHER"
    return "UNKNOWN"

def build_dataset(
    model_name: str,
    dataset_name: str = "stanfordnlp/SHP",
    split: str = "test",
    input_min_text_length: int = 2,
    input_max_text_length: int = 512,
):
    """
    EXACTLY the same SHP handling you used for PPO:
      - prompt is sample["history"]
      - filter on len(sample["history"])
      - tokenization uses max_length=256 (fixed)
      - stores "input_ids" and "query"
      - sets format to torch with columns ["input_ids","query"]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, split=split)

    ds = ds.filter(
        lambda x: input_min_text_length <= len(x["history"]) <= input_max_text_length
    )

    def tokenize(sample):
        prompt = sample["history"]
        input_ids = tokenizer.encode(
            prompt,
            truncation=True,
            max_length=256,
        )
        sample["input_ids"] = input_ids
        sample["query"] = tokenizer.decode(input_ids, skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch", columns=["input_ids", "query"])
    return ds

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="../config/llmevaluate.yaml", type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)

    accelerator = Accelerator()
    rank = accelerator.process_index
    device = accelerator.device

    log_dir = cfg.get("logging", {}).get("dir", "/efs/rohandeb/logs/qwen/shp/dpo_eval")
    logger = setup_logging(log_dir, rank=rank)
    logger.info("Starting DPO checkpoint evaluation on SHP (standard vs curriculum)")
    logger.info(json.dumps(cfg, indent=2, default=str))

    dataset_name = cfg["dataset"]["name"]
    split        = cfg["dataset"]["split"]
    model_name   = cfg["model_name"]
    batch_size   = int(cfg["training"]["batch_size"])

    max_new_tokens = int(cfg["dataset"]["max_response_length"])
    max_prompt_length = int(cfg["dataset"]["max_prompt_length"])  # kept for config parity; SHP tokenization fixed at 256

    MODEL_DIR_standard = cfg["checkpoints"]["dir_standard"]
    MODEL_DIR_curr     = cfg["checkpoints"]["dir_curr"]
    comparison_dir     = cfg["checkpoints"]["save_dir"]
    os.makedirs(comparison_dir, exist_ok=True)

    # Outputs (per-rank)
    results_txt_path = os.path.join(comparison_dir, f"results_rank{rank}.txt")
    results_jsonl_path = os.path.join(comparison_dir, f"comparisons_rank{rank}.jsonl")
    results_txt = open(results_txt_path, "a", buffering=1)
    results_jsonl = open(results_jsonl_path, "a", buffering=1)

    # Dataset: EXACTLY PPO SHP handling
    dataset = build_dataset(
        model_name=model_name,
        dataset_name=dataset_name,
        split=split,
        input_min_text_length=int(cfg.get("dataset", {}).get("input_min_text_length", 2)),
        input_max_text_length=int(cfg.get("dataset", {}).get("input_max_text_length", 512)),
    )

    if cfg["dataset"].get("max_samples"):
        dataset = dataset.select(range(int(cfg["dataset"]["max_samples"])))
    N = len(dataset)

    logger.info(f"effective_max_samples_loaded={N}")
    logger.info(f"results_txt_path={results_txt_path}")
    logger.info(f"results_jsonl_path={results_jsonl_path}")
    logger.info(f"(note) SHP tokenization max_length fixed at 256 per PPO eval; cfg max_prompt_length={max_prompt_length}")

    if accelerator.is_main_process:
        meta = {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "split": split,
            "N_loaded_after_selection": N,
            "model_name": model_name,
            "MODEL_DIR_standard": MODEL_DIR_standard,
            "MODEL_DIR_curr": MODEL_DIR_curr,
            "world_size": accelerator.num_processes,
            "config_path": args.config,
        }
        with open(os.path.join(comparison_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # Models (DPO checkpoints = plain CausalLM)
    logger.info(f"Loading DPO models:\n  standard={MODEL_DIR_standard}\n  curr={MODEL_DIR_curr}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR_standard,
        local_files_only=True,
        trust_remote_code=True,
    ).eval().to(device)

    model_curr = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR_curr,
        local_files_only=True,
        trust_remote_code=True,
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    generation_kwargs = {
        "do_sample": True,
        "top_k": 0,
        "top_p": 1.0,
        "pad_token_id": tokenizer.eos_token_id,
        "min_new_tokens": 8,
        "max_new_tokens": max_new_tokens,
    }

    # Counts: [A, B, NEITHER]
    local_counts = torch.zeros(3, dtype=torch.long, device=device)
    local_unknown = torch.zeros(1, dtype=torch.long, device=device)

    try:
        all_indices = list(range(N))
        with accelerator.split_between_processes(all_indices) as local_indices:
            it = local_indices
            if accelerator.is_local_main_process:
                it = tqdm(local_indices)

            with torch.no_grad():
                for idx_chunk in chunk_list(list(it), batch_size):
                    batch_items = [dataset[i] for i in idx_chunk]
                    prompts = [x["query"] for x in batch_items]
                    query_tensors = [x["input_ids"].to(device) for x in batch_items]
                    input_lengths = [int(t.numel()) for t in query_tensors]

                    # Pad to batch for generate
                    batch_enc = tokenizer.pad(
                        {"input_ids": query_tensors},
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    attention_mask = batch_enc.get("attention_mask", None)

                    # Standard model generations
                    gen_std = model.generate(
                        input_ids=batch_enc["input_ids"],
                        attention_mask=attention_mask,
                        **generation_kwargs,
                    )

                    # Curriculum model generations
                    gen_cur = model_curr.generate(
                        input_ids=batch_enc["input_ids"],
                        attention_mask=attention_mask,
                        **generation_kwargs,
                    )

                    # Slice off prompt portion per-example (using original unpadded lengths)
                    generated_texts = []
                    generated_texts_curr = []
                    for out_ids, out_ids_cur, in_len in zip(gen_std, gen_cur, input_lengths):
                        resp_ids = out_ids[in_len:]
                        resp_ids_cur = out_ids_cur[in_len:]
                        generated_texts.append(tokenizer.decode(resp_ids, skip_special_tokens=True))
                        generated_texts_curr.append(tokenizer.decode(resp_ids_cur, skip_special_tokens=True))

                    # Judge + save
                    for global_idx, ctx, a, b in zip(idx_chunk, prompts, generated_texts, generated_texts_curr):
                        record = {
                            "timestamp": datetime.now().isoformat(),
                            "rank": int(rank),
                            "idx": int(global_idx),
                            "context": ctx,
                            "response_A": a,
                            "response_B": b,
                        }

                        try:
                            verdict_raw = evaluate_pair(ctx, a, b)
                            record["verdict_raw"] = verdict_raw
                        except Exception as e:
                            logger.exception(f"Judge call failed on rank {rank} idx {global_idx}: {e}")
                            local_unknown[0] += 1
                            record["choice"] = "UNKNOWN"
                            record["error"] = str(e)
                            results_jsonl.write(json.dumps(record) + "\n")
                            continue

                        choice = parse_first_bold_choice(verdict_raw)
                        record["choice"] = choice

                        verdict_wrapped = "\n".join(wrap(verdict_raw, width=100))
                        logger.info(f"idx={global_idx} choice={choice}\n{verdict_wrapped}\n")

                        # txt
                        results_txt.write(f"idx={global_idx} choice={choice}\n")
                        results_txt.write(f"Context:\n{ctx}\n\n")
                        results_txt.write(f"A:\n{a}\n\n")
                        results_txt.write(f"B:\n{b}\n\n")
                        results_txt.write(f"Claude verdict:\n{verdict_wrapped}\n")
                        results_txt.write("\n" + ("=" * 100) + "\n\n")

                        # jsonl
                        results_jsonl.write(json.dumps(record) + "\n")

                        if choice == "A":
                            local_counts[0] += 1
                        elif choice == "B":
                            local_counts[1] += 1
                        elif choice == "NEITHER":
                            local_counts[2] += 1
                        else:
                            local_unknown[0] += 1

        accelerator.wait_for_everyone()

        g_counts = accelerator.gather(local_counts)
        g_unknown = accelerator.gather(local_unknown)

        if g_counts.ndim == 1:
            g_counts = g_counts.view(accelerator.num_processes, -1)  # (world_size, 3)
        if g_unknown.ndim == 2:
            g_unknown = g_unknown.view(-1)  # (world_size,)

        if accelerator.is_main_process:
            totals = g_counts.sum(dim=0).cpu().tolist()  # [A, B, NEITHER]
            total_unknown = int(g_unknown.sum().item())

            A_cnt, B_cnt, N_cnt = int(totals[0]), int(totals[1]), int(totals[2])
            judged = A_cnt + B_cnt + N_cnt

            denom = float(judged) if judged > 0 else 1.0
            A_pct = 100.0 * A_cnt / denom
            B_pct = 100.0 * B_cnt / denom
            N_pct = 100.0 * N_cnt / denom

            logger.info(f"TOTAL COUNTS (judged): A={A_cnt}, B={B_cnt}, Neither={N_cnt}, Judged={judged}")
            logger.info(f"PERCENTAGES (of judged): A_win={A_pct:.2f}%, B_win={B_pct:.2f}%, Draws={N_pct:.2f}%")
            logger.info(f"UNKNOWN / PARSE / JUDGE FAILURES (excluded from percentages): {total_unknown}")
            logger.info(f"TOTAL LOADED (max_samples after selection) = {N}")

            summary_path = os.path.join(comparison_dir, "summary.txt")
            with open(summary_path, "a", buffering=1) as f:
                f.write(f"timestamp={datetime.now().isoformat()}\n")
                f.write(f"TOTAL LOADED (max_samples after selection) = {N}\n")
                f.write(f"TOTAL COUNTS (judged): A={A_cnt}, B={B_cnt}, Neither={N_cnt}, Judged={judged}\n")
                f.write(f"PERCENTAGES (of judged): A_win={A_pct:.2f}%, B_win={B_pct:.2f}%, Draws={N_pct:.2f}%\n")
                f.write(f"UNKNOWN / PARSE / JUDGE FAILURES (excluded from percentages): {total_unknown}\n")
                f.write("\n")

    finally:
        try:
            results_txt.close()
        except Exception:
            pass
        try:
            results_jsonl.close()
        except Exception:
            pass

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
