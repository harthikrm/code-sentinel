.PHONY: install lint format baseline train evaluate deploy help

help:
	@echo "Code Sentinel — available targets: install lint format baseline train evaluate deploy"

install:
	@echo "TODO: create venv and pip install -r requirements.txt"

lint:
	@echo "TODO: run ruff check ."

format:
	@echo "TODO: run ruff format . and black ."

baseline:
	@echo "TODO: run training/baseline_eval.py"

train:
	@echo "TODO: run training/train.py"

evaluate:
	@echo "TODO: run evaluation/compare_models.py"

deploy:
	@echo "TODO: run serving/vertex_deploy.py"
