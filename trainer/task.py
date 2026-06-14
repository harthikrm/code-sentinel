# Vertex AI entrypoint — loads experiment config and runs QLoRA training.

import json
import os

from trainer.train import train


def main() -> None:
    """
    Start fine-tuning on Vertex AI.

    Reads TRAINING_CONFIG_JSON from the environment (set by vertex_deploy.py).
    Falls back to run1_config for local smoke tests.
    """
    config_json = os.environ.get("TRAINING_CONFIG_JSON")
    if config_json:
        config = json.loads(config_json)
    else:
        from trainer.config import run1_config

        config = run1_config

    # Vertex writes artifacts to AIP_MODEL_DIR when base_output_dir is set.
    model_dir = os.environ.get("AIP_MODEL_DIR")
    if model_dir:
        config = dict(config)
        config["output_dir"] = model_dir

    train(config)


if __name__ == "__main__":
    main()
