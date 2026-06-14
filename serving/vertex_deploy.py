# Submit Code Sentinel QLoRA training jobs to Google Vertex AI custom training.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.auth
from google.api_core import exceptions as gcp_exceptions
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import aiplatform, storage

# -----------------------------------------------------------------------------
# Vertex AI project settings
# -----------------------------------------------------------------------------
PROJECT_ID = "code-sentinel-499017"
REGIONS = ["us-east4", "europe-west4", "us-central1"]
STAGING_BUCKET = "gs://code-sentinel-training"
# Vertex rejects multi-region buckets for job staging/output — use per-region buckets.
REGION_BUCKETS = {
    "us-east4": "gs://code-sentinel-training-use4",
    "europe-west4": "gs://code-sentinel-training-euw4",
    "us-central1": "gs://code-sentinel-training",
}
# .py310 images are required for Python package training on Vertex.
BASE_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
PYTHON_MODULE = "trainer.task"

# GPU profiles — set config["gpu_profile"] to "t4" or "a100", or rely on smoke_test default.
GPU_PROFILES: Dict[str, Dict[str, Any]] = {
    "t4": {
        "machine_type": "n1-standard-8",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
    },
    "h100": {
        "machine_type": "a3-highgpu-1g",
        "accelerator_type": "NVIDIA_H100_80GB",
        "accelerator_count": 1,
    },
}
DEFAULT_GPU_PROFILE = "h100"
QUOTA_CONSOLE_URL = (
    f"https://console.cloud.google.com/iam-admin/quotas"
    f"?project={PROJECT_ID}&pageState=(%22allQuotasTable%22"
    f":(%22f%22:%22%255B%255D%22))"
)

# Installed via setup.py when Vertex pip-installs the training package.
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
    "sentencepiece",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
TRAINER_DIR = REPO_ROOT / "trainer"
PACKAGE_NAME = "code-sentinel-trainer"
PACKAGE_VERSION = "0.1.0"


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


def _is_capacity_error(exc: BaseException) -> bool:
    """Return True if the error indicates regional GPU quota or capacity exhaustion."""
    if isinstance(exc, gcp_exceptions.ResourceExhausted):
        return True
    msg = str(exc).lower()
    capacity_markers = (
        "capacity",
        "quota",
        "resource exhausted",
        "resourceexhausted",
        "insufficient",
        "not available in region",
        "no capacity",
    )
    return any(marker in msg for marker in capacity_markers)


def _format_quota_error(
    gpu: Dict[str, Any],
    exc: BaseException,
    *,
    regions: Optional[List[str]] = None,
) -> str:
    """Return actionable guidance when Vertex GPU quota is exhausted in all regions."""
    tried = regions or REGIONS
    regions_list = ", ".join(tried)
    return (
        f"Vertex AI GPU quota exceeded for profile '{gpu['name']}' "
        f"({gpu['accelerator_type']}) in all tried regions: {regions_list}.\n\n"
        "Your GCP project likely has 0 training GPU quota in these regions. Request an increase:\n"
        f"  1. Open: {QUOTA_CONSOLE_URL}\n"
        f"  2. Filter for regions: {regions_list}\n"
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

    Priority: config["gpu_profile"] > VERTEX_GPU_PROFILE env > DEFAULT_GPU_PROFILE (a100).
    """
    profile_name = (
        config.get("gpu_profile")
        or os.environ.get("VERTEX_GPU_PROFILE")
        or DEFAULT_GPU_PROFILE
    )
    if profile_name not in GPU_PROFILES:
        valid = ", ".join(sorted(GPU_PROFILES))
        raise ValueError(f"Unknown gpu_profile '{profile_name}'. Choose one of: {valid}")

    profile = dict(GPU_PROFILES[profile_name])
    profile["name"] = profile_name
    return profile


def _validate_config(config: Dict[str, Any]) -> str:
    """Validate experiment config and return the run name."""
    run_name = config.get("run_name")
    if not run_name:
        raise ValueError("config must include 'run_name' to identify the experiment.")
    return str(run_name)


def _sync_trainer_sources() -> None:
    """
    Refresh trainer/train.py and trainer/config.py from training/ before packaging.

    Keeps a single source of truth in training/ while shipping a self-contained
    trainer package to Vertex.
    """
    if not TRAINER_DIR.exists():
        raise FileNotFoundError(f"Trainer package directory not found: {TRAINER_DIR}")

    shutil.copy2(TRAINING_DIR / "train.py", TRAINER_DIR / "train.py")
    shutil.copy2(TRAINING_DIR / "config.py", TRAINER_DIR / "config.py")


def _write_setup_py(package_root: Path) -> None:
    """Write setuptools config so Vertex can pip-install the trainer package."""
    requirements_literal = ",\n        ".join(f'"{req}"' for req in TRAINING_REQUIREMENTS)
    setup_py = f'''\
from setuptools import find_packages, setup

setup(
    name="{PACKAGE_NAME}",
    version="{PACKAGE_VERSION}",
    packages=find_packages(),
    install_requires=[
        {requirements_literal}
    ],
    description="Code Sentinel QLoRA training package for Vertex AI.",
)
'''
    (package_root / "setup.py").write_text(setup_py, encoding="utf-8")


def _build_source_distribution(staging_dir: Path) -> Path:
    """
    Build a setuptools source distribution containing the full trainer/ package.

    Layout:
      setup.py
      trainer/
        __init__.py
        task.py
        train.py
        config.py
    """
    _sync_trainer_sources()

    package_root = staging_dir / "package_root"
    shutil.copytree(TRAINER_DIR, package_root / "trainer")
    _write_setup_py(package_root)

    cmd = [sys.executable, "setup.py", "sdist", "--formats=gztar"]
    result = subprocess.run(
        cmd,
        cwd=package_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build trainer source distribution.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    dist_dir = package_root / "dist"
    archives = sorted(dist_dir.glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No .tar.gz produced in {dist_dir}")

    return archives[-1]


def _upload_package_to_gcs(local_archive: Path, run_name: str) -> str:
    """Upload the trainer tar.gz to GCS and return its gs:// URI."""
    bucket_name = STAGING_BUCKET.replace("gs://", "")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    blob_name = f"packages/{run_name}/{PACKAGE_NAME}-{timestamp}.tar.gz"

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(str(local_archive))

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"Uploaded training package to {gcs_uri}")
    return gcs_uri


def _build_environment_variables(
    config: Dict[str, Any],
    wandb_api_key: Optional[str] = None,
) -> Dict[str, str]:
    """Build env vars passed to the Vertex worker via python_package_spec.env."""
    api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "WANDB_API_KEY must be set in the environment or passed to launch_training_job()."
        )

    return {
        "TRAINING_CONFIG_JSON": json.dumps(config),
        "WANDB_API_KEY": api_key,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def _build_worker_pool_specs(
    package_gcs_uri: str,
    gpu: Dict[str, Any],
    environment_variables: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Build CustomJob worker pool specs using python_package_spec.

    This is the correct Vertex pattern for multi-file Python training code:
    pip-install the uploaded package, then run `python -m trainer.task`.
    """
    return [
        {
            "machine_spec": {
                "machine_type": gpu["machine_type"],
                "accelerator_type": gpu["accelerator_type"],
                "accelerator_count": gpu["accelerator_count"],
            },
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri": BASE_IMAGE,
                "package_uris": [package_gcs_uri],
                "python_module": PYTHON_MODULE,
                "env": [
                    {"name": key, "value": value}
                    for key, value in environment_variables.items()
                ],
            },
        }
    ]


def launch_training_job(
    config: Dict[str, Any],
    *,
    wandb_api_key: Optional[str] = None,
    sync: bool = False,
) -> aiplatform.CustomJob:
    """
    Package the trainer/ Python package and submit a Vertex AI CustomJob.

    Args:
        config: Experiment dictionary from training/config.py (must include run_name).
        wandb_api_key: Optional W&B API key; defaults to WANDB_API_KEY env var.
        sync: If True, block until training finishes. If False, return once the
            CustomJob resource is created.

    Returns:
        The submitted google.cloud.aiplatform.CustomJob instance.
    """
    run_name = _validate_config(config)
    _check_gcp_credentials()
    gpu = _resolve_gpu_profile(config)

    job_config = dict(config)
    job_config.setdefault(
        "output_dir",
        f"{STAGING_BUCKET}/checkpoints/{run_name}",
    )
    environment_variables = _build_environment_variables(job_config, wandb_api_key=wandb_api_key)

    with tempfile.TemporaryDirectory(prefix="code-sentinel-vertex-") as tmp:
        staging_dir = Path(tmp)
        archive_path = _build_source_distribution(staging_dir)
        package_gcs_uri = _upload_package_to_gcs(archive_path, run_name)

        worker_pool_specs = _build_worker_pool_specs(
            package_gcs_uri=package_gcs_uri,
            gpu=gpu,
            environment_variables=environment_variables,
        )

        display_name = f"code-sentinel-{run_name}"

        job: Optional[aiplatform.CustomJob] = None
        selected_region: Optional[str] = None
        selected_base_output_dir: Optional[str] = None
        last_capacity_error: Optional[BaseException] = None

        for region in REGIONS:
            region_bucket = REGION_BUCKETS[region]
            base_output_dir = f"{region_bucket}/vertex-jobs/{run_name}"
            print(f"Submitting Vertex AI job in region: {region} (staging: {region_bucket})")
            aiplatform.init(
                project=PROJECT_ID,
                location=region,
                staging_bucket=region_bucket,
            )

            job = aiplatform.CustomJob(
                display_name=display_name,
                worker_pool_specs=worker_pool_specs,
                base_output_dir=base_output_dir,
                project=PROJECT_ID,
                location=region,
                staging_bucket=region_bucket,
            )

            try:
                if sync:
                    job.run(sync=True)
                else:
                    job.run(sync=False)
                    job.wait_for_resource_creation()
                selected_region = region
                selected_base_output_dir = base_output_dir
                print(f"Job submitted successfully in {region}")
                break
            except Exception as exc:
                if _is_capacity_error(exc):
                    last_capacity_error = exc
                    print(
                        f"Capacity/quota error in {region}: {exc}\n"
                        "Trying next region..."
                    )
                    continue
                raise

        if selected_region is None or job is None:
            raise RuntimeError(
                _format_quota_error(gpu, last_capacity_error or RuntimeError("Unknown error"), regions=REGIONS)
            ) from last_capacity_error

        print(f"Submitted Vertex AI training job: {display_name}")
        print(f"  run_name:       {run_name}")
        print(f"  region:         {selected_region}")
        print(f"  python_module:  {PYTHON_MODULE}")
        print(f"  package_uri:    {package_gcs_uri}")
        print(f"  gpu_profile:    {gpu['name']} ({gpu['machine_type']} + {gpu['accelerator_type']})")
        print(f"  output_dir:     {job_config['output_dir']}")
        print(f"  base_output:    {selected_base_output_dir}")
        print(f"  resource_name:  {job.resource_name}")
        print(
            "  console:        "
            f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
            f"?project={PROJECT_ID}&region={selected_region}"
        )

        return job


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    from training.config import run1_config, run8_config
    launch_training_job(run1_config)
    launch_training_job(run8_config)
