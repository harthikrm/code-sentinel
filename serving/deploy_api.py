# Build, publish, and deploy the Code Sentinel FastAPI service on Vertex AI.

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.auth
from google.api_core import exceptions as gcp_exceptions
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import aiplatform

# -----------------------------------------------------------------------------
# Vertex AI / Artifact Registry settings
# -----------------------------------------------------------------------------
PROJECT_ID = "code-sentinel-2026"
REGION = "us-central1"
MODEL_PATH = "gs://code-sentinel-2026-training/merged-model/run1"
ARTIFACT_REGISTRY_HOST = f"{REGION}-docker.pkg.dev"
REPOSITORY = "code-sentinel"
IMAGE_NAME = "api"
IMAGE_TAG = "latest"
IMAGE_URI = "us-central1-docker.pkg.dev/code-sentinel-2026/code-sentinel/api:latest"

DEFAULT_MODEL_PATH = "gs://code-sentinel-2026-training/merged-model/run1"

# Online-prediction GPU profiles (separate from *training* GPU quota in Vertex).
# Training H100 quota does not apply to endpoint serving — request
# "Custom model serving NVIDIA H100 GPUs" if you need H100 inference.
GPU_PROFILES: Dict[str, Dict[str, Any]] = {
    "a100": {
        "machine_type": "a2-highgpu-1g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 1,
    },
    "l4": {
        "machine_type": "g2-standard-4",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 1,
    },
    "h100": {
        "machine_type": "a3-highgpu-1g",
        "accelerator_type": "NVIDIA_H100_80GB",
        "accelerator_count": 1,
    },
}
# Prefer A100/L4 for serving — H100 serving quota is often 0 or already consumed.
DEFAULT_GPU_PROFILE = "a100"
GPU_PROFILE_FALLBACK_ORDER = ["a100", "l4", "h100"]

DEPLOY_MAX_RETRIES = 3
DEPLOY_RETRY_DELAY_SECONDS = 30
DEPLOY_REQUEST_TIMEOUT_SECONDS = 3600.0
MIN_REPLICA_COUNT = 1
MAX_REPLICA_COUNT = 1

MODEL_DISPLAY_NAME = "code-sentinel-api"
ENDPOINT_DISPLAY_NAME = "code-sentinel-api-endpoint"
DEPLOYED_MODEL_DISPLAY_NAME = "code-sentinel-api"

CONTAINER_PORT = 8080
HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "serving" / "Dockerfile"


def _check_gcp_credentials() -> None:
    """
    Verify Application Default Credentials before Docker push or Vertex deploy.

    Raises:
        DefaultCredentialsError: If ADC is not configured.
    """
    try:
        google.auth.default()
    except DefaultCredentialsError as exc:
        raise DefaultCredentialsError(
            "Google Cloud credentials not found. Set up one of the following:\n\n"
            "Option A — gcloud CLI (recommended for local dev):\n"
            "  gcloud auth application-default login\n"
            "  gcloud auth configure-docker us-central1-docker.pkg.dev\n"
            "  gcloud config set project code-sentinel-2026\n\n"
            "Option B — service account key:\n"
            "  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json\n"
        ) from exc


def _run_command(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    """
    Run a shell command and raise if it exits non-zero.

    Args:
        cmd: Command argv (no shell interpolation).
        cwd: Optional working directory.

    Raises:
        RuntimeError: If the command fails.
    """
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if result.stdout.strip():
        print(result.stdout.strip())


def _configure_docker_auth() -> None:
    """Configure Docker credential helper for Artifact Registry."""
    _run_command(
        ["gcloud", "auth", "configure-docker", f"{ARTIFACT_REGISTRY_HOST}", "--quiet"]
    )


def build_container(*, image_uri: str = IMAGE_URI) -> str:
    """
    Build the Code Sentinel API Docker image from ``serving/Dockerfile``.

    The build context is the repository root so ``requirements.txt`` and
    ``data/utils.py`` are available to the Dockerfile.

    Args:
        image_uri: Tag applied to the built image.

    Returns:
        The image URI that was built.
    """
    if not DOCKERFILE.is_file():
        raise FileNotFoundError(f"Dockerfile not found: {DOCKERFILE}")

    _run_command(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "-f",
            str(DOCKERFILE),
            "-t",
            image_uri,
            str(REPO_ROOT),
        ]
    )
    print(f"Built image: {image_uri}")
    return image_uri


def push_container(*, image_uri: str = IMAGE_URI) -> str:
    """
    Push a locally built image to Google Artifact Registry.

    Args:
        image_uri: Fully qualified image URI to push.

    Returns:
        The pushed image URI.
    """
    _configure_docker_auth()
    _run_command(["docker", "push", image_uri])
    print(f"Pushed image: {image_uri}")
    return image_uri


def upload_model(
    *,
    image_uri: str,
    model_path: str = DEFAULT_MODEL_PATH,
    display_name: str = MODEL_DISPLAY_NAME,
) -> aiplatform.Model:
    """
    Register the container image as a Vertex AI Model resource.

    The model container exposes ``/health`` for probes and ``/predict`` for
    Vertex online prediction (``instances`` / ``predictions`` JSON envelope).

    Args:
        image_uri: Artifact Registry URI for the serving container.
        model_path: GCS or local path passed to the app via ``MODEL_PATH``.
        display_name: Vertex AI model display name.

    Returns:
        The uploaded ``aiplatform.Model`` instance.
    """
    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.Model.upload(
        display_name=display_name,
        serving_container_image_uri=image_uri,
        serving_container_ports=[CONTAINER_PORT],
        serving_container_environment_variables={"MODEL_PATH": model_path},
        serving_container_health_route=HEALTH_ROUTE,
        serving_container_predict_route=PREDICT_ROUTE,
        sync=True,
    )
    print(f"Uploaded Vertex AI model: {model.resource_name}")
    print(f"  MODEL_PATH: {model_path}")
    return model


def _resolve_gpu_profile(profile_name: str) -> Dict[str, Any]:
    """Return machine/accelerator settings for a named GPU profile."""
    if profile_name not in GPU_PROFILES:
        valid = ", ".join(sorted(GPU_PROFILES))
        raise ValueError(f"Unknown gpu_profile '{profile_name}'. Choose one of: {valid}")
    profile = dict(GPU_PROFILES[profile_name])
    profile["name"] = profile_name
    return profile


def _format_serving_quota_error(exc: BaseException, profile: Dict[str, Any]) -> str:
    """Return actionable guidance when Vertex *serving* GPU quota is exhausted."""
    return (
        f"Vertex AI *serving* GPU quota exceeded for profile '{profile['name']}' "
        f"({profile['accelerator_type']}) in {REGION}.\n\n"
        "Training GPU quota and serving GPU quota are separate in GCP.\n"
        "Your 2× H100 training quota does not apply to online prediction endpoints.\n\n"
        "Options:\n"
        "  1. Free serving GPUs — undeploy old endpoint models:\n"
        f"       python serving/deploy_api.py --list-deployments\n"
        f"       python serving/deploy_api.py --undeploy-existing\n"
        "  2. Try another serving GPU profile:\n"
        f"       python serving/deploy_api.py --skip-build --skip-push --gpu-profile a100\n"
        f"       python serving/deploy_api.py --skip-build --skip-push --gpu-profile l4\n"
        "  3. Request quota increase (IAM → Quotas → filter us-central1):\n"
        "     - Custom model serving NVIDIA A100 GPUs\n"
        "     - Custom model serving NVIDIA L4 GPUs\n"
        "     - Custom model serving NVIDIA H100 GPUs\n\n"
        f"API error: {exc}"
    )


def list_endpoint_deployments() -> None:
    """Print all Vertex endpoints and deployed models that consume serving GPUs."""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoints = aiplatform.Endpoint.list(order_by="create_time desc")
    if not endpoints:
        print("No Vertex endpoints found.")
        return

    for endpoint in endpoints:
        print(f"\nEndpoint: {endpoint.display_name}")
        print(f"  resource: {endpoint.resource_name}")
        deployed_models = endpoint.list_models()
        if not deployed_models:
            print("  (no deployed models)")
            continue
        for deployed in deployed_models:
            print(f"  - {deployed.display_name} (id={deployed.id})")


def undeploy_existing_models(
    *,
    endpoint_display_name: str = ENDPOINT_DISPLAY_NAME,
) -> None:
    """
    Undeploy all models from the Code Sentinel endpoint to free serving GPUs.

    Args:
        endpoint_display_name: Endpoint display name to clean up.
    """
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc",
    )
    if not endpoints:
        print(f"No endpoint named '{endpoint_display_name}' found.")
        return

    endpoint = endpoints[0]
    deployed_models = endpoint.list_models()
    if not deployed_models:
        print(f"No deployed models on {endpoint.resource_name}")
        return

    for deployed in deployed_models:
        print(f"Undeploying {deployed.display_name} (id={deployed.id})...")
        endpoint.undeploy(deployed_model_id=deployed.id, sync=True)
    print("Done — serving GPUs released.")


def _get_or_create_endpoint(
    *,
    display_name: str = ENDPOINT_DISPLAY_NAME,
) -> aiplatform.Endpoint:
    """
    Return an existing Vertex endpoint or create a new one.

    Args:
        display_name: Endpoint display name to match or create.

    Returns:
        An ``aiplatform.Endpoint`` ready for deployment.
    """
    aiplatform.init(project=PROJECT_ID, location=REGION)

    existing = aiplatform.Endpoint.list(
        filter=f'display_name="{display_name}"',
        order_by="create_time desc",
    )
    if existing:
        endpoint = existing[0]
        print(f"Using existing endpoint: {endpoint.resource_name}")
        return endpoint

    endpoint = aiplatform.Endpoint.create(display_name=display_name, sync=True)
    print(f"Created endpoint: {endpoint.resource_name}")
    return endpoint


def deploy_endpoint(
    model: aiplatform.Model,
    *,
    endpoint: Optional[aiplatform.Endpoint] = None,
    gpu_profile: str = DEFAULT_GPU_PROFILE,
    deployed_model_display_name: str = DEPLOYED_MODEL_DISPLAY_NAME,
    sync: bool = True,
    try_fallback_profiles: bool = True,
) -> aiplatform.Endpoint:
    """
    Deploy a registered model to a Vertex AI endpoint.

    Uses GPU profiles suitable for Mistral-7B inference. Tries fallback profiles
    when serving GPU quota is exhausted (training quota is separate).

    Args:
        model: Uploaded Vertex AI model resource.
        endpoint: Optional existing endpoint; created if omitted.
        gpu_profile: Profile name from ``GPU_PROFILES`` or ``auto``.
        deployed_model_display_name: Name shown on the endpoint deployment.
        sync: Block until deployment completes.
        try_fallback_profiles: Try other profiles after quota errors when ``auto``.

    Returns:
        The endpoint with the new deployment attached.
    """
    aiplatform.init(project=PROJECT_ID, location=REGION)

    if endpoint is None:
        endpoint = _get_or_create_endpoint()

    if gpu_profile == "auto":
        profiles_to_try = [_resolve_gpu_profile(name) for name in GPU_PROFILE_FALLBACK_ORDER]
    else:
        profiles_to_try = [_resolve_gpu_profile(gpu_profile)]

    last_error: Optional[BaseException] = None

    for profile_index, profile in enumerate(profiles_to_try):
        deploy_kwargs = {
            "model": model,
            "deployed_model_display_name": deployed_model_display_name,
            "machine_type": profile["machine_type"],
            "accelerator_type": profile["accelerator_type"],
            "accelerator_count": profile["accelerator_count"],
            "min_replica_count": MIN_REPLICA_COUNT,
            "max_replica_count": MAX_REPLICA_COUNT,
            "sync": sync,
            "deploy_request_timeout": DEPLOY_REQUEST_TIMEOUT_SECONDS,
        }
        print(
            f"Deploying with gpu_profile={profile['name']} "
            f"({profile['machine_type']} + {profile['accelerator_type']})"
        )

        for attempt in range(1, DEPLOY_MAX_RETRIES + 1):
            try:
                endpoint.deploy(**deploy_kwargs)
                print(f"Deployed model to endpoint: {endpoint.resource_name}")
                print(f"  gpu_profile:  {profile['name']}")
                print(f"  machine_type: {profile['machine_type']} + {profile['accelerator_type']}")
                print(
                    "  console:      "
                    f"https://console.cloud.google.com/vertex-ai/online-prediction/endpoints"
                    f"?project={PROJECT_ID}&region={REGION}"
                )
                return endpoint
            except gcp_exceptions.ResourceExhausted as exc:
                last_error = exc
                if attempt < DEPLOY_MAX_RETRIES:
                    print(
                        f"Quota error (attempt {attempt}/{DEPLOY_MAX_RETRIES}): {exc}\n"
                        f"Retrying in {DEPLOY_RETRY_DELAY_SECONDS}s..."
                    )
                    time.sleep(DEPLOY_RETRY_DELAY_SECONDS)
                    continue
                break
            except gcp_exceptions.GoogleAPICallError as exc:
                last_error = exc
                if attempt < DEPLOY_MAX_RETRIES:
                    print(
                        f"Deploy attempt {attempt}/{DEPLOY_MAX_RETRIES} failed: {exc}\n"
                        f"Retrying in {DEPLOY_RETRY_DELAY_SECONDS}s..."
                    )
                    time.sleep(DEPLOY_RETRY_DELAY_SECONDS)
                    continue
                raise

        if try_fallback_profiles and gpu_profile == "auto" and profile_index < len(profiles_to_try) - 1:
            print(_format_serving_quota_error(last_error or RuntimeError("unknown"), profile))
            print("Trying next GPU profile...\n")
            continue

        if last_error:
            raise RuntimeError(_format_serving_quota_error(last_error, profile)) from last_error

    raise RuntimeError("Deploy failed with no GPU profile available.")


def deploy_api(
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    image_uri: str = IMAGE_URI,
    skip_build: bool = False,
    skip_push: bool = False,
    gpu_profile: str = "auto",
    undeploy_existing: bool = False,
) -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
    """
    Build, push, and deploy the Code Sentinel FastAPI service end-to-end.

    Steps:
      1. Build Docker image from ``serving/Dockerfile``
      2. Push to Artifact Registry
      3. Upload as a Vertex AI Model with ``MODEL_PATH`` set
      4. Deploy to a Vertex AI Endpoint (A100 → L4 → H100 fallback by default)

    Args:
        model_path: GCS URI of merged model weights for the container.
        image_uri: Artifact Registry destination tag.
        skip_build: Skip ``docker build`` (reuse local image).
        skip_push: Skip ``docker push`` (image already in registry).
        gpu_profile: ``auto``, ``a100``, ``l4``, or ``h100``.
        undeploy_existing: Undeploy models on the endpoint before deploying.

    Returns:
        Tuple of ``(model, endpoint)``.
    """
    _check_gcp_credentials()

    if undeploy_existing:
        undeploy_existing_models()

    if not skip_build:
        build_container(image_uri=image_uri)
    if not skip_push:
        push_container(image_uri=image_uri)

    model = upload_model(image_uri=image_uri, model_path=model_path)
    endpoint = deploy_endpoint(model, gpu_profile=gpu_profile)
    return model, endpoint


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI flags for ``deploy_api``."""
    parser = argparse.ArgumentParser(
        description="Build and deploy the Code Sentinel FastAPI service to Vertex AI."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"GCS/local model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--image-uri",
        default=IMAGE_URI,
        help=f"Artifact Registry image URI (default: {IMAGE_URI})",
    )
    parser.add_argument(
        "--gpu-profile",
        default="auto",
        choices=["auto", *sorted(GPU_PROFILES)],
        help="Serving GPU profile (default: auto tries a100 → l4 → h100)",
    )
    parser.add_argument(
        "--undeploy-existing",
        action="store_true",
        help="Undeploy all models on the endpoint before deploying (frees serving GPUs).",
    )
    parser.add_argument(
        "--list-deployments",
        action="store_true",
        help="List Vertex endpoints and deployed models, then exit.",
    )
    parser.add_argument(
        "--undeploy-only",
        action="store_true",
        help="Undeploy existing endpoint models and exit (free serving GPUs).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip docker build (reuse local image).",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip docker push (image already in registry).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    _check_gcp_credentials()

    if args.list_deployments:
        list_endpoint_deployments()
        raise SystemExit(0)

    if args.undeploy_only:
        undeploy_existing_models()
        raise SystemExit(0)

    deploy_api(
        model_path=args.model_path,
        image_uri=args.image_uri,
        skip_build=args.skip_build,
        skip_push=args.skip_push,
        gpu_profile=args.gpu_profile,
        undeploy_existing=args.undeploy_existing,
    )
