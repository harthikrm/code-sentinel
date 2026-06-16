# Code Sentinel

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Vertex AI](https://img.shields.io/badge/Vertex%20AI-Custom%20Training-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/vertex-ai)
[![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-Logging-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)

**Fine-tuned Mistral-7B-Instruct-v0.3 for automated, actionable code review on real pull-request diffs.**

**Repository:** https://github.com/harthikrm/code-sentinel

---

## Problem

Engineering teams at scale review hundreds of PRs daily. Senior engineers who catch subtle bugs, security vulnerabilities, and architectural issues are bottlenecked. Generic LLMs do surface-level review but don't understand what experienced engineers actually catch or how they phrase feedback. Code Sentinel fine-tunes on 150K real reviewer comments to learn those patterns.

---

## Dataset

**Source:** [Microsoft CodeReviewer](https://zenodo.org/record/7650861) on Zenodo (~6.9 GB raw). Each example contains a code diff (hunk), the full file context, and a human reviewer comment across six languages: Python, Java, JavaScript, C#, C++, and Go.

**Quality filters applied:**

- Minimum 20-character comment length
- Removed bot comments: URLs (`http`, `https`, `www`), static-analysis rule codes (e.g. `E501` via `[A-Z]\d+`), tool references (e.g. `[flake8]`, `[pylint]`)
- English only

**Final split (from 169,058 total filtered examples):**

| Split | Examples |
|-------|----------|
| Train | 143,996 |
| Validation | 12,545 |
| Test | 12,517 |

Processed JSONL files: `data/processed-train.jsonl`, `data/processed-valid.jsonl`, `data/processed-test.jsonl`. GCS mirror: `gs://code-sentinel-training-us/data/`.

Each example is formatted as a Mistral `[INST]` prompt pairing language, diff hunk, and reviewer comment (`data/preprocess.py`).

---

## Approach

**Base model:** [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

**QLoRA fine-tuning:**

- 4-bit NF4 quantization via `bitsandbytes` — reduces the ~14 GB model to ~4 GB VRAM at load time
- LoRA adapters (r = 16, α = 32) on attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Only ~0.8% of parameters trained; base weights frozen in 4-bit

**Training:**

- Hugging Face TRL `SFTTrainer` on instruction-formatted examples (`[INST] … [/INST] {comment}`)
- Cosine LR schedule, warmup ratio 0.03, gradient checkpointing
- NVIDIA H100 on Google Vertex AI (`serving/vertex_deploy.py`)
- Metrics and hyperparameters logged in Weights & Biases

**Deployment:**

- LoRA adapters merged into base weights (`serving/merge_adapters.py`)
- FastAPI inference service (`serving/api.py`) on Vertex AI

---

## Experiments

Hyperparameter search over LoRA rank, learning rate, and epoch count. Configurations in `training/config.py`.

| Run | Rank | LR | Epochs | Train Loss | Eval Loss |
|-----|------|----|--------|------------|-----------|
| run1 | 16 | 2e-4 | 1 | [TBD] | [TBD] |
| run3 | 8 | 2e-4 | 3 | [TBD] | [TBD] |
| run4 | 32 | 2e-4 | 3 | [TBD] | [TBD] |
| run5 | 16 | 1e-4 | 3 | [TBD] | [TBD] |
| run8 | 64 | 2e-4 | 1 | [TBD] | [TBD] |

Submit a training job:

```bash
python serving/vertex_deploy.py
```

---

## Results

BERTScore F1 on the held-out test set (English):

| Model | BERTScore F1 |
|-------|--------------|
| Base Mistral 7B | 0.7149 |
| Fine-tuned Code Sentinel (run1, r=16, lr=2e-4) | 0.7470 |
| GPT-4o-mini | 0.7041 |

Evaluation: `evaluation/compare_models.py`.

---

## Key Findings

- Fine-tuned Mistral-7B outperforms base model by 4.5% BERTScore F1 (0.7470 vs 0.7149)
- Fine-tuned model outperforms GPT-4o-mini by 6.1% (0.7470 vs 0.7041)
- Lower LoRA rank (r=8, run3) achieved best eval loss among 3-epoch runs (~1.32)
- Higher rank (r=32, run4) showed higher eval loss and instability (~1.38)

---

## Failure Analysis

Coming soon — experiments in progress.

---

## How to Use

```bash
pip install -r requirements.txt

export MODEL_PATH=gs://code-sentinel-training-us/merged-model/run1
export OPENAI_API_KEY=your_key  # for comparison only

uvicorn serving.api:app --host 0.0.0.0 --port 8080
```

```bash
curl -X POST http://localhost:8080/review \
  -H "Content-Type: application/json" \
  -d '{"diff": "+ def foo():\n+   pass", "lang": "py"}'
```

**Training / cloud (optional):**

| Variable | Purpose |
|----------|---------|
| `WANDB_API_KEY` | Experiment logging on Vertex AI |
| `HF_TOKEN` | Push adapters to Hugging Face Hub |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCS and Vertex AI access |

---

## References

1. Dettmers, T., et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs.** https://arxiv.org/abs/2305.14314
2. Hu, E. J., et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models.** https://arxiv.org/abs/2106.09685
3. Li, Z., et al. (2022). **CodeReviewer: Pre-Training for Automating Code Review Activities.** https://arxiv.org/abs/2203.09095
4. Jiang, A. Q., et al. (2023). **Mistral 7B.** https://arxiv.org/abs/2310.06825
