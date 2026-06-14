# Build, publish, and deploy the Code Sentinel FastAPI service on Vertex AI.

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import google.auth
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import aiplatform

# -----------------------------------------------------------------------------
# Vertex AI / Artifact Registry settings
# -----------------------------------------------------------------------------
PROJECT_ID = "code-sentinel-499017"
REGION = "us-central1"
ARTIFACT_REGISTRY_HOST = f"{REGION}-docker.pkg.dev"
REPOSITORY = "code-sentinel"
IMAGE_NAME = "api"
IMAGE_TAG = "latest"
IMAGE_URI = f"{ARTIFACT_REGISTRY_HOST}/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:{IMAGE_TAG}"

DEFAULT_MODEL_PATH = "gs://code-sentinel-training-us/merged-model/run1"
MACHINE_TYPE = "n1-standard-4"
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
            "  gcloud config set project code-sentinel-499017\n\n"
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
    machine_type: str = MACHINE_TYPE,
    deployed_model_display_name: str = DEPLOYED_MODEL_DISPLAY_NAME,
    sync: bool = True,
) -> aiplatform.Endpoint:
    """
    Deploy a registered model to a Vertex AI endpoint.

    Uses a CPU ``n1-standard-4`` worker by default (no GPU accelerators).

    Args:
        model: Uploaded Vertex AI model resource.
        endpoint: Optional existing endpoint; created if omitted.
        machine_type: Compute machine type for inference replicas.
        deployed_model_display_name: Name shown on the endpoint deployment.
        sync: Block until deployment completes.

    Returns:
        The endpoint with the new deployment attached.
    """
    aiplatform.init(project=PROJECT_ID, location=REGION)

    if endpoint is None:
        endpoint = _get_or_create_endpoint()

    endpoint.deploy(
        model=model,
        deployed_model_display_name=deployed_model_display_name,
        machine_type=machine_type,
        min_replica_count=MIN_REPLICA_COUNT,
        max_replica_count=MAX_REPLICA_COUNT,
        sync=sync,
    )
    print(f"Deployed model to endpoint: {endpoint.resource_name}")
    print(f"  machine_type: {machine_type}")
    print(
        "  console:      "
        f"https://console.cloud.google.com/vertex-ai/online-prediction/endpoints"
        f"?project={PROJECT_ID}&region={REGION}"
    )
    return endpoint


def deploy_api(
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    image_uri: str = IMAGE_URI,
    skip_build: bool = False,
    skip_push: bool = False,
    machine_type: str = MACHINE_TYPE,
) -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
    """
    Build, push, and deploy the Code Sentinel FastAPI service end-to-end.

    Steps:
      1. Build Docker image from ``serving/Dockerfile``
      2. Push to ``us-central1-docker.pkg.dev/code-sentinel-499017/code-sentinel/api:latest``
      3. Upload as a Vertex AI Model with ``MODEL_PATH`` set
      4. Deploy to a Vertex AI Endpoint on ``n1-standard-4`` (CPU)

    Args:
        model_path: GCS URI of merged model weights for the container.
        image_uri: Artifact Registry destination tag.
        skip_build: Skip ``docker build`` (reuse local image).
        skip_push: Skip ``docker push`` (image already in registry).
        machine_type: Inference machine type.

    Returns:
        Tuple of ``(model, endpoint)``.
    """
    _check_gcp_credentials()

    if not skip_build:
        build_container(image_uri=image_uri)
    if not skip_push:
        push_container(image_uri=image_uri)

    model = upload_model(image_uri=image_uri, model_path=model_path)
    endpoint = deploy_endpoint(model, machine_type=machine_type)
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
        "--machine-type",
        default=MACHINE_TYPE,
        help=f"Inference machine type (default: {MACHINE_TYPE})",
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
    deploy_api(
        model_path=args.model_path,
        image_uri=args.image_uri,
        skip_build=args.skip_build,
        skip_push=args.skip_push,
        machine_type=args.machine_type,
    )
