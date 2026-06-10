# Code Sentinel — QLoRA training entrypoint (Mistral-7B + W&B + BERTScore).

import os
from typing import Any, Dict, Optional

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

def train(config: Dict[str, Any]) -> None:
    """
    Fine-tune `mistralai/Mistral-7B-Instruct-v0.3` using QLoRA + TRL SFTTrainer.

    Expected `config` keys (based on `training/config.py`):
      - LoRA params: r, lora_alpha, target_modules, lora_dropout (optional), bias, task_type
      - Training params: learning_rate, num_train_epochs, per_device_train_batch_size,
        gradient_accumulation_steps, warmup_ratio, lr_scheduler_type, logging_steps,
        eval_steps, save_steps, fp16, output_dir, run_name
      - Data params (recommended): train_data_path, validation_data_path
      - Optional: wandb_project, bertscore_lang, bertscore_num_samples

    The training/validation JSONL must be line-delimited JSON objects with a `text` field.
    """

    if torch.cuda.is_available():
        device_type = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    # fp16 is CUDA-only per constraint. On MPS we avoid fp16.
    use_fp16 = bool(config.get("fp16", True)) and device_type == "cuda"
    compute_dtype = torch.float16 if device_type == "cuda" else torch.float32

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    train_path = config.get("train_data_path", "./data/processed-train.jsonl")
    validation_path = config.get("validation_data_path", "./data/processed-valid.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"train_data_path not found: {train_path}. "
            "Set config['train_data_path'] to your processed JSONL file."
        )
    if not os.path.exists(validation_path):
        raise FileNotFoundError(
            f"validation_data_path not found: {validation_path}. "
            "Set config['validation_data_path'] to your processed JSONL file."
        )

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Initialize Weights & Biases
    # -----------------------------
    wandb_project = config.get("wandb_project", "code-sentinel")
    run_name = config["run_name"]
    wandb.init(project=wandb_project, name=run_name, config=config)

    class WandbLossAndEvalCallback(TrainerCallback):
        """Forward trainer logs/evals to W&B on each logging/eval step."""

        def __init__(self) -> None:
            super().__init__()
            self._last_train_loss: Optional[float] = None
            self._last_learning_rate: Optional[float] = None

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            if not logs:
                return
            step = state.global_step
            payload: Dict[str, Any] = {}
            if "loss" in logs:
                self._last_train_loss = float(logs["loss"])
                payload["train_loss"] = self._last_train_loss
            if "learning_rate" in logs:
                self._last_learning_rate = float(logs["learning_rate"])
                payload["learning_rate"] = self._last_learning_rate
            if payload:
                wandb.log(payload, step=step)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
            if not metrics:
                return
            step = state.global_step
            payload: Dict[str, Any] = {}

            # Requirement: include (latest) train loss alongside eval loss + BERTScore.
            if self._last_train_loss is not None:
                payload["train_loss"] = self._last_train_loss
            if self._last_learning_rate is not None:
                payload["learning_rate"] = self._last_learning_rate

            # HF typically prefixes metrics with `eval_`. Keep names as-is for clarity.
            for k in ["eval_loss", "eval_bertscore_f1", "eval_bertscore_precision", "eval_bertscore_recall"]:
                if k in metrics and metrics[k] is not None:
                    payload[k] = float(metrics[k])

            # Fall back: log any bertscore-* fields if naming differs.
            if not payload:
                bert_fields = {k: v for k, v in metrics.items() if "bertscore" in k and isinstance(v, (int, float))}
                if bert_fields:
                    payload.update({k: float(v) for k, v in bert_fields.items()})

            if payload:
                wandb.log(payload, step=step)

    # -----------------------------
    # Load dataset from processed JSONL
    # -----------------------------
    data_files = {"train": train_path, "validation": validation_path}
    dataset = load_dataset("json", data_files=data_files)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # -----------------------------
    # Load tokenizer + base model in 4-bit NF4
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        # Mistral checkpoints often have no explicit pad token.
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Attempt 4-bit NF4 load; on some Apple Silicon setups bitsandbytes may not support MPS.
    # In that case we fall back to a non-quantized load so the pipeline can still run.
    # On MPS, avoid device_map="auto" — it can split layers across meta and mps:0 and break backprop.
    try:
        if device_type == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
            )
            model = model.to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
    except Exception as e:
        if device_type == "mps":
            print(
                "WARNING: 4-bit NF4 load failed on MPS. "
                "Falling back to non-quantized loading for local dev."
            )
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model = model.to("mps")
        else:
            raise RuntimeError("Failed to load model with 4-bit NF4 quantization.") from e
    model.config.use_cache = False

    # -----------------------------
    # Attach LoRA adapters
    # -----------------------------
    task_type_str = config.get("task_type", "CAUSAL_LM")
    task_type = TaskType.CAUSAL_LM

    lora_config = LoraConfig(
        r=int(config["r"]),
        lora_alpha=int(config["lora_alpha"]),
        target_modules=list(config["target_modules"]),
        lora_dropout=float(config.get("lora_dropout", 0.0)),
        bias=config.get("bias", "none"),
        task_type=task_type,
    )
    model = get_peft_model(model, lora_config)

    # -----------------------------
    # Compute BERTScore during evaluation
    # -----------------------------
    def compute_metrics(eval_pred) -> Dict[str, float]:
        """
        Compute BERTScore on eval batches.

        Notes:
          - For causal LM evaluation, HF may provide logits. We argmax to token ids.
          - References are decoded from `label_ids` (with -100 positions replaced by pad token).
        """

        try:
            from bert_score import score as bertscore_score
        except ImportError as e:
            raise ImportError(
                "Missing dependency `bert-score`. Add it to requirements.txt to compute BERTScore."
            ) from e

        preds = getattr(eval_pred, "predictions", eval_pred[0])
        label_ids = getattr(eval_pred, "label_ids", eval_pred[1])

        # Some trainers return a tuple for predictions.
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        # Convert logits -> token ids if needed.
        if preds.ndim == 3:
            pred_token_ids = preds.argmax(axis=-1)
        else:
            pred_token_ids = preds

        # Replace ignored positions (-100) in labels.
        pad_id = tokenizer.pad_token_id
        label_ids = torch.as_tensor(label_ids)
        label_ids = torch.where(label_ids == -100, torch.tensor(pad_id, device=label_ids.device), label_ids)

        # Decode
        pred_token_ids = torch.as_tensor(pred_token_ids)
        pred_texts = tokenizer.batch_decode(
            pred_token_ids[: config.get("bertscore_num_samples", 32)].cpu().tolist(),
            skip_special_tokens=True,
        )
        ref_texts = tokenizer.batch_decode(
            label_ids[: config.get("bertscore_num_samples", 32)].cpu().tolist(),
            skip_special_tokens=True,
        )

        lang = config.get("bertscore_lang", "en")
        P, R, F1 = bertscore_score(pred_texts, ref_texts, lang=lang, verbose=False)

        return {
            "bertscore_precision": float(P.mean().detach().cpu().item()),
            "bertscore_recall": float(R.mean().detach().cpu().item()),
            "bertscore_f1": float(F1.mean().detach().cpu().item()),
        }

    # -----------------------------
    # Configure and train with SFTTrainer
    # -----------------------------
    max_seq_length = 2048

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=max_seq_length,
        num_train_epochs=int(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]),
        lr_scheduler_type=config["lr_scheduler_type"],
        logging_steps=int(config["logging_steps"]),
        eval_steps=int(config["eval_steps"]),
        save_steps=int(config["save_steps"]),
        eval_strategy="steps",
        save_strategy="steps",
        dataset_text_field="text",
        fp16=use_fp16,
        bf16=False,
        report_to=[],  # we manually log to W&B via callbacks
        prediction_loss_only=False,
        seed=config.get("seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[WandbLossAndEvalCallback()],
    )

    try:
        trainer.train()
        # Save adapter weights (PEFT) to the configured output dir.
        trainer.model.save_pretrained(output_dir)
    finally:
        wandb.finish()


if __name__ == "__main__":
    from config import run1_config

    train(run1_config)
