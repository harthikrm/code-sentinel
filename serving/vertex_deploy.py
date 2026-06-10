# Submit Code Sentinel QLoRA training jobs to Google Vertex AI custom training.

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.auth
from google.api_core import exceptions as gcp_exceptions
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import aiplatform

# -----------------------------------------------------------------------------
# Vertex AI project settings
# -----------------------------------------------------------------------------
PROJECT_ID = "code-sentinel-499017"
REGION = "us-central1"
STAGING_BUCKET = "gs://code-sentinel-training"
# .py310 images are required for from_local_script / Python package training.
BASE_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"

# GPU profiles — set config["gpu_profile"] to "t4" or "a100", or rely on smoke_test default.
GPU_PROFILES: Dict[str, Dict[str, Any]] = {
    "t4": {
        "machine_type": "n1-standard-8",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
    },
    "a100": {
        "machine_type": "a2-highgpu-1g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 1,
    },
}
DEFAULT_GPU_PROFILE = "t4"
QUOTA_CONSOLE_URL = (
    f"https://console.cloud.google.com/iam-admin/quotas"
    f"?project={PROJECT_ID}&pageState=(%22allQuotasTable%22"
    f":(%22f%22:%22%255B%255D%22))"
)

# Python packages installed on the training VM at job start.
TRAINING_REQUIREMENTS: List[str] = [
    "transformers==4.57.6",
    "peft",
    "trl",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "wandb",
    "bert-score",
    "gcsfs",
]

# Path to training sources relative to the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
TRAIN_SCRIPT = TRAINING_DIR / "train.py"
CONFIG_MODULE = TRAINING_DIR / "config.py"

# Worker entrypoint written into the staged package before submission.
VERTEX_ENTRYPOINT = "vertex_entry.py"
VERTEX_ENTRYPOINT_SOURCE = '''\
# Vertex AI worker entrypoint — loads config from env and calls train().
import json
import os

from train import train


def main() -> None:
    """Deserialize training config from TRAINING_CONFIG_JSON and start fine-tuning."""
    config_json = os.environ.get("TRAINING_CONFIG_JSON")
    if not config_json:
        raise RuntimeError("TRAINING_CONFIG_JSON environment variable is required.")

    config = json.loads(config_json)

    # Vertex AI exposes AIP_MODEL_DIR for writing job artifacts to Cloud Storage.
    model_dir = os.environ.get("AIP_MODEL_DIR")
    if model_dir:
        config["output_dir"] = model_dir

    train(config)


if __name__ == "__main__":
    main()
'''


def _check_gcp_credentials() -> None:
    """
    Verify Application Default Credentials are available before uploading to GCS.

    Raises:
        DefaultCredentialsError: With setup instructions if credentials are missing.
    """
    try:
        google.auth.default()
    except DefaultCredentialsError as exc:
        raise DefaultCredentialsError(
            "Google Cloud credentials not found. Set up one of the following:\n\n"
            "Option A — gcloud CLI (recommended for local dev):\n"
            "  brew install --cask google-cloud-sdk\n"
            "  gcloud auth application-default login\n"
            "  gcloud config set project code-sentinel-499017\n\n"
            "Option B — service account key:\n"
            "  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json\n"
        ) from exc


def _format_quota_error(gpu: Dict[str, Any], exc: gcp_exceptions.ResourceExhausted) -> str:
    """Return actionable guidance when Vertex GPU quota is exhausted."""
    return (
        f"Vertex AI GPU quota exceeded for profile '{gpu['name']}' "
        f"({gpu['accelerator_type']}).\n\n"
        "Your GCP project likely has 0 training GPU quota. Request an increase:\n"
        f"  1. Open: {QUOTA_CONSOLE_URL}\n"
        f"  2. Filter for region: {REGION}\n"
        "  3. Search and request increases for:\n"
        "     - Custom model training NVIDIA T4 GPUs\n"
        "     - Custom model training NVIDIA A100 GPUs (for full runs)\n"
        "  4. Set requested value to at least 1 and submit justification\n"
        "     (e.g. 'QLoRA fine-tuning Mistral-7B for code review research').\n\n"
        "Approval can take hours to a few business days.\n"
        "While waiting, you can run smoke tests locally: python training/train.py\n\n"
        f"API error: {exc}"
    )


def _resolve_gpu_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pick a Vertex machine/accelerator profile for this run.

    Priority: config["gpu_profile"] > VERTEX_GPU_PROFILE env > smoke_test heuristic.
    Smoke tests default to T4 (widely available); full runs default to A100.
    """
    profile_name = (
        config.get("gpu_profile")
        or os.environ.get("VERTEX_GPU_PROFILE")
        or ("t4" if config.get("smoke_test") else DEFAULT_GPU_PROFILE)
    )
    if profile_name not in GPU_PROFILES:
        valid = ", ".join(sorted(GPU_PROFILES))
        raise ValueError(f"Unknown gpu_profile '{profile_name}'. Choose one of: {valid}")

    profile = dict(GPU_PROFILES[profile_name])
    profile["name"] = profile_name
    return profile


def _validate_config(config: Dict[str, Any]) -> str:
    """
    Validate the experiment config and return the run name used to label the job.

    Raises:
        ValueError: If `run_name` is missing from the config dictionary.
    """
    run_name = config.get("run_name")
    if not run_name:
        raise ValueError("config must include 'run_name' to identify the experiment.")
    return str(run_name)


def _prepare_training_package(staging_dir: Path) -> Path:
    """
    Copy training modules into a temporary package directory for Vertex staging.

    Vertex packages the parent directory of `script_path`, so we place `train.py`,
    `config.py`, and a small entrypoint script in the same folder.
    """
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")
    if not CONFIG_MODULE.exists():
        raise FileNotFoundError(f"Training config module not found: {CONFIG_MODULE}")

    package_dir = staging_dir / "training_pkg"
    package_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(TRAIN_SCRIPT, package_dir / "train.py")
    shutil.copy2(CONFIG_MODULE, package_dir / "config.py")

    entrypoint_path = package_dir / VERTEX_ENTRYPOINT
    entrypoint_path.write_text(VERTEX_ENTRYPOINT_SOURCE, encoding="utf-8")

    return entrypoint_path


def _build_environment_variables(
    config: Dict[str, Any],
    wandb_api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build environment variables passed to the Vertex training worker.

    TRAINING_CONFIG_JSON carries the full experiment config.
    WANDB_API_KEY enables Weights & Biases logging inside train.py.
    """
    api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "WANDB_API_KEY must be set in the environment or passed to launch_training_job()."
        )

    # Serialize once so the worker can reconstruct the exact experiment config.
    env = {
        "TRAINING_CONFIG_JSON": json.dumps(config),
        "WANDB_API_KEY": api_key,
    }
    return env


def launch_training_job(
    config: Dict[str, Any],
    *,
    wandb_api_key: Optional[str] = None,
    sync: bool = False,
) -> aiplatform.CustomJob:
    """
    Package `training/train.py` and submit a Vertex AI custom training job.

    Args:
        config: Experiment dictionary from `training/config.py` (must include `run_name`).
        wandb_api_key: Optional W&B API key; defaults to the local `WANDB_API_KEY` env var.
        sync: If True, block until training finishes. If False, return once the
            CustomJob is created (does not wait for training to complete).

    Returns:
        The submitted `google.cloud.aiplatform.CustomJob` instance.
    """
    run_name = _validate_config(config)
    _check_gcp_credentials()
    gpu = _resolve_gpu_profile(config)

    # Use a copy so we can route checkpoints to GCS without mutating caller state.
    job_config = dict(config)
    job_config.setdefault(
        "output_dir",
        f"{STAGING_BUCKET}/checkpoints/{run_name}",
    )

    environment_variables = _build_environment_variables(job_config, wandb_api_key=wandb_api_key)

    # Initialize the Vertex SDK for this project/region and staging bucket.
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=STAGING_BUCKET,
    )

    display_name = f"code-sentinel-{run_name}"
    base_output_dir = f"{STAGING_BUCKET}/vertex-jobs/{run_name}"

    with tempfile.TemporaryDirectory(prefix="code-sentinel-vertex-") as tmp:
        staging_dir = Path(tmp)
        entrypoint_path = _prepare_training_package(staging_dir)

        # Submit a custom training job using the pre-built PyTorch GPU container.
        # Requirements are installed on the VM before the entrypoint script runs.
        job = aiplatform.CustomJob.from_local_script(
            display_name=display_name,
            script_path=str(entrypoint_path),
            container_uri=BASE_IMAGE,
            requirements=TRAINING_REQUIREMENTS,
            project=PROJECT_ID,
            location=REGION,
            staging_bucket=STAGING_BUCKET,
            machine_type=gpu["machine_type"],
            accelerator_type=gpu["accelerator_type"],
            accelerator_count=gpu["accelerator_count"],
            base_output_dir=base_output_dir,
            environment_variables=environment_variables,
        )

        # from_local_script() only builds the job; run() submits it to Vertex AI.
        # sync=False launches in a background thread — we must wait for resource
        # creation before the process exits, otherwise the job is never submitted.
        try:
            if sync:
                job.run(sync=True)
            else:
                job.run(sync=False)
                job.wait_for_resource_creation()
        except gcp_exceptions.ResourceExhausted as exc:
            raise gcp_exceptions.ResourceExhausted(
                _format_quota_error(gpu, exc)
            ) from exc

        print(f"Submitted Vertex AI training job: {display_name}")
        print(f"  run_name:      {run_name}")
        print(f"  gpu_profile:   {gpu['name']} ({gpu['machine_type']} + {gpu['accelerator_type']})")
        print(f"  output_dir:    {job_config['output_dir']}")
        print(f"  base_output:   {base_output_dir}")
        print(f"  resource_name: {job.resource_name}")
        print(
            "  console:       "
            f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
            f"?project={PROJECT_ID}&region={REGION}"
        )

        return job


if __name__ == "__main__":
    # Allow `python serving/vertex_deploy.py` to resolve `training.config`.
    sys.path.insert(0, str(REPO_ROOT))

    from training.config import run1_config

    launch_training_job(run1_config)
