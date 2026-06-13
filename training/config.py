"""
This document contains training configurations for 10 experiements we will be performing to fine the model.
Runs 9 and 10 are updated once we find the best configurations from runs 1-8 in wandb.
"""

run1_config = {"r" : 16,
               "lora_alpha" : 32,
               "target_modules" : ["q_proj", "v_proj", "k_proj", "o_proj"],
               "lora_dropout" : 0.05,
               "bias" : "none",
               "task_type" : "CAUSAL_LM",
               "learning_rate" : 2e-4,
               "num_train_epochs" : 1,
               "per_device_train_batch_size" : 1,
               "per_device_eval_batch_size" : 1,
               "gradient_accumulation_steps" : 16,
               "max_seq_length" : 1024,
               "warmup_ratio" : 0.03,
               "lr_scheduler_type" : "cosine",
               "output_dir": "gs://code-sentinel-training/checkpoints/run1",
               "logging_steps": 50,
               "eval_steps": 500,
               "save_steps": 500,
               "fp16" : True,
               "run_name" : "run1-r16-lr2e4-1epoch",
               "hf_repo_id": "harthikrm/code-sentinel-run1",
               "train_data_path": "gs://code-sentinel-training/data/processed-train.jsonl",
               "validation_data_path": "gs://code-sentinel-training/data/processed-valid.jsonl"}

run2_config = {**run1_config, 
              "num_train_epochs" : 3,
              "output_dir" : "./checkpoints/run2",
              "run_name" : "run2-r16-lr2e4-3epoch"}

subset_config = {**run2_config,
                 "train_data_path": "gs://code-sentinel-training/data/processed-train.jsonl",
                 "max_train_samples": 20000,
                 "output_dir": "gs://code-sentinel-training/checkpoints/subset",
                 "run_name": "subset-r16-lr2e4-20k-3epoch"}

run3_config = {**run2_config,
               "r" : 8,
               "output_dir" : "gs://code-sentinel-training/checkpoints/run3",
               "run_name" : "run3-r8-lr2e4-3epoch"}

run4_config = {**run2_config,
               "r" : 32,
               "output_dir" : "gs://code-sentinel-training/checkpoints/run4",
               "run_name" : "run4-r32-lr2e4-3epoch"}

run5_config = {**run2_config,
               "learning_rate" : 1e-4,
               "output_dir" : "gs://code-sentinel-training/checkpoints/run5",
               "run_name" : "run5-r16-lr1e4-3epoch"}

run6_config = {**run2_config,
               "learning_rate" : 3e-4,
               "output_dir" : "./checkpoints/run6",
               "run_name" : "run6-r16-lr3e4-3epoch"}

run7_config = {**run2_config,
               "r" : 16,
               "learning_rate" : 2e-4,
               "output_dir" : "./checkpoints/run7",
               "lang_filter" : "py",
               "run_name" : "run7-r16-lr2e4-py-only"}

run8_config = {**run2_config,
               "r" : 64,
               "output_dir" : "./checkpoints/run8",
               "run_name" : "run8-r64-lr2e4-3epoch"}

# run9 and run10 — update best_config after analyzing runs 1-8 in W&B

run9_config = {**run2_config,
               "lora_dropout" : 0.1,
               "output_dir" : "./checkpoints/run9",
               "run_name" : "run9-r16-lr2e4-3epoch"}

run10_config = {}
