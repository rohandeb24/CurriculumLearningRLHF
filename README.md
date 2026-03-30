# CurriculumLearning

RLHF training with curriculum learning strategies using `Qwen/Qwen2-0.5B`. Experiments run across two datasets (**HH-RLHF** and **Stanford SHP**) and two algorithms (**DPO** and **PPO**).

---

## Curriculum Strategies

| Directory | Description | Algorithms | Datasets |
|---|---|---|---|
| `baseline_tuning` | Standard training, no curriculum | DPO, PPO | HH-RLHF, SHP |
| `entropy` | Curriculum ordered by entropy of the preference signal | DPO, PPO | HH-RLHF, SHP |
| `length` | Curriculum ordered by response length | DPO, PPO | HH-RLHF, SHP |
| `clustering` | Curriculum ordered by embedding cluster difficulty | DPO, PPO | HH-RLHF, SHP |
| `ensemble` | Curriculum difficulty estimated via bootstrap reward model ensemble | PPO | HH-RLHF, SHP |
| `pec` | Priority Experience Curriculum using TD/advantage scores | PPO | HH-RLHF, SHP |

---

## Repository Structure

```
CurriculumLearning/
├── baseline_tuning/
│   ├── hh-rlhf/{dpo,ppo}/
│   │   ├── config/config.yaml
│   │   └── src/{dpo,ppo}.py
│   └── stanfordshp/{dpo,ppo}/  (same layout)
├── entropy/
│   ├── hh-rlhf/{dpo,ppo}/
│   │   ├── config/config.yaml
│   │   └── src/{dpo,ppo}_increasing.py
│   └── stanfordshp/{dpo,ppo}/  (same layout)
├── length/         (same layout as entropy)
├── clustering/     (same layout as entropy)
├── ensemble/
│   ├── hh-rlhf/ppo/   (PPO only)
│   └── stanfordshp/ppo/
├── pec/
│   ├── hh-rlhf/ppo/   (PPO only)
│   └── stanfordshp/ppo/
└── plots/
    ├── histogram.py
    ├── histogram_hh_dpo.py
    └── histogram_stanford_ppo.py
```

Each experiment directory contains an `LLMEvaluate.py` script for LLM-based comparison; run outputs go to `checkpoints.save_dir` in `llmevaluate.yaml` (typically under `models/.../llmevaluate/`).

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate rlhf
```

> Use the `trl` version from `environment.yml` (currently `0.8.0`). Newer `trl` releases change `DPOTrainer` to require `DPOConfig` instead of `TrainingArguments`, and `0.12+` removes the legacy `PPOTrainer` / `PPOConfig` stack. Install `trl` with `pip install --no-deps 'trl==0.8.0'` so it does not upgrade `torch` / `transformers`.

---

## Running (Multi-GPU)

All scripts are launched with `accelerate launch`. Replace `<N>` with the number of GPUs available.

### Baseline Tuning

```bash
# HH-RLHF — DPO
accelerate launch --multi_gpu --num_processes=1 \
  baseline_tuning/hh-rlhf/dpo/src/dpo.py \
  -c baseline_tuning/hh-rlhf/dpo/config/config.yaml

# HH-RLHF — PPO
accelerate launch --multi_gpu --num_processes=1 \
  baseline_tuning/hh-rlhf/ppo/src/ppo.py \
  -c baseline_tuning/hh-rlhf/ppo/config/config.yaml

# Stanford SHP — DPO
accelerate launch --multi_gpu --num_processes=<N> \
  baseline_tuning/stanfordshp/dpo/src/dpo.py \
  -c baseline_tuning/stanfordshp/dpo/config/config.yaml

# Stanford SHP — PPO
accelerate launch --multi_gpu --num_processes=<N> \
  baseline_tuning/stanfordshp/ppo/src/ppo.py \
  -c baseline_tuning/stanfordshp/ppo/config/config.yaml
```

### Entropy Curriculum

```bash
# HH-RLHF — DPO
accelerate launch --multi_gpu --num_processes=1 \
  entropy/hh-rlhf/dpo/src/dpo_increasing.py \
  -c entropy/hh-rlhf/dpo/config/config.yaml

# HH-RLHF — PPO
accelerate launch --multi_gpu --num_processes=<N> \
  entropy/hh-rlhf/ppo/src/ppo_increasing.py \
  -c entropy/hh-rlhf/ppo/config/config.yaml

# Stanford SHP — DPO
accelerate launch --multi_gpu --num_processes=<N> \
  entropy/stanfordshp/dpo/src/dpo_increasing.py \
  -c entropy/stanfordshp/dpo/config/config.yaml

# Stanford SHP — PPO
accelerate launch --multi_gpu --num_processes=<N> \
  entropy/stanfordshp/ppo/src/ppo_increasing.py \
  -c entropy/stanfordshp/ppo/config/config.yaml
```

#### Entropy PPO dataset (`scored_rewards_dataset_with_entropy`)

The **entropy PPO** scripts load the dataset via `datasets.load_from_disk()` (a local on-disk HuggingFace dataset), so `dataset.name` in:
- `entropy/hh-rlhf/ppo/config/config.yaml`
- `entropy/stanfordshp/ppo/config/config.yaml`

must point to a directory on disk that contains the dataset (named `scored_rewards_dataset_with_entropy/...`).

For entropy curriculum, this saved dataset should include an `entropy` column computed per preference pair using:

```text
z = beta * ((log pi_theta(chosen) - log pi_theta(rejected)) - (log pi_ref(chosen) - log pi_ref(rejected)))
p = sigmoid(z)
entropy = -p * log(p) - (1 - p) * log(1 - p)
```


### Length Curriculum

```bash
# HH-RLHF — DPO
accelerate launch --multi_gpu --num_processes=<N> \
  length/hh-rlhf/dpo/src/dpo_increasing.py \
  -c length/hh-rlhf/dpo/config/config.yaml

# HH-RLHF — PPO
accelerate launch --multi_gpu --num_processes=<N> \
  length/hh-rlhf/ppo/src/ppo_increasing.py \
  -c length/hh-rlhf/ppo/config/config.yaml

# Stanford SHP — DPO
accelerate launch --multi_gpu --num_processes=<N> \
  length/stanfordshp/dpo/src/dpo_increasing.py \
  -c length/stanfordshp/dpo/config/config.yaml

# Stanford SHP — PPO
accelerate launch --multi_gpu --num_processes=<N> \
  length/stanfordshp/ppo/src/ppo_increasing.py \
  -c length/stanfordshp/ppo/config/config.yaml
```

### Clustering Curriculum

```bash
# HH-RLHF — DPO
accelerate launch --multi_gpu --num_processes=<N> \
  clustering/hh-rlhf/dpo/src/dpo_increasing.py \
  -c clustering/hh-rlhf/dpo/config/config.yaml

# HH-RLHF — PPO
accelerate launch --multi_gpu --num_processes=<N> \
  clustering/hh-rlhf/ppo/src/ppo_increasing.py \
  -c clustering/hh-rlhf/ppo/config/config.yaml

# Stanford SHP — DPO
accelerate launch --multi_gpu --num_processes=<N> \
  clustering/stanfordshp/dpo/src/dpo_increasing.py \
  -c clustering/stanfordshp/dpo/config/config.yaml

# Stanford SHP — PPO
accelerate launch --multi_gpu --num_processes=<N> \
  clustering/stanfordshp/ppo/src/ppo_increasing.py \
  -c clustering/stanfordshp/ppo/config/config.yaml
```

### Ensemble Curriculum

```bash
# HH-RLHF — PPO
accelerate launch --multi_gpu --num_processes=<N> \
  ensemble/hh-rlhf/ppo/src/ppo_increasing.py \
  -c ensemble/hh-rlhf/ppo/config/config.yaml

# Stanford SHP — PPO
accelerate launch --multi_gpu --num_processes=<N> \
  ensemble/stanfordshp/ppo/src/ppo_increasing.py \
  -c ensemble/stanfordshp/ppo/config/config.yaml
```

### PEC (Priority Experience Curriculum)

```bash
# HH-RLHF — PPO
accelerate launch --multi_gpu --num_processes=<N> \
  pec/hh-rlhf/ppo/src/ppo_increasing.py \
  -c pec/hh-rlhf/ppo/config/config.yaml

# Stanford SHP — PPO
accelerate launch --multi_gpu --num_processes=<N> \
  pec/stanfordshp/ppo/src/ppo_increasing.py \
  -c pec/stanfordshp/ppo/config/config.yaml
```

### LLM Evaluation (all strategies)

Each strategy directory contains a `LLMEvaluate.py` script. Before running, set the API-related constants at the top of that file (around lines 20–22): `CLAUDE_MODEL_ID`, `OPENAI_ENDPOINT`, and `API_KEY`.

Also update the checkpoint directory paths in the strategy's `llmevaluate.yaml` (specifically the `checkpoints.dir_standard` and `checkpoints.dir_curr` fields; for example, in `ensemble/hh-rlhf/ppo/config/llmevaluate.yaml` this is around lines 43–44). These should point to your trained `standard` and `curr` checkpoints.

Example:

```bash
accelerate launch --multi_gpu --num_processes=<N> \
  entropy/hh-rlhf/ppo/src/LLMEvaluate.py \
  -c entropy/hh-rlhf/ppo/config/llmevaluate.yaml
```

---

