# Merge QLoRA adapter weights into the base Mistral model for standalone deployment.

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch
from google.cloud import storage
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER_PATH = "./checkpoints/run1"
DEFAULT_OUTPUT_PATH = "gs://code-sentinel-training/merged-model/run1"


def _is_gcs_path(path: str) -> bool:
    """Return True if the path is a Google Cloud Storage URI."""
    return path.startswith("gs://")


def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    """Split gs://bucket/prefix into (bucket_name, blob_prefix)."""
    if not _is_gcs_path(uri):
        raise ValueError(f"Not a GCS URI: {uri}")
    without_scheme = uri[len("gs://") :]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix.rstrip("/")


def _download_gcs_prefix(gcs_uri: str, local_dir: Path) -> Path:
    """Download all blobs under a GCS prefix into a local directory."""
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket, prefix=prefix))

    if not blobs:
        raise FileNotFoundError(f"No objects found at {gcs_uri}")

    local_dir.mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(prefix) + 1 :] if prefix else blob.name
        destination = local_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))

    return local_dir


def _upload_directory_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    """Upload a local directory tree to a GCS prefix."""
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(local_dir).as_posix()
        blob_name = f"{prefix}/{relative}" if prefix else relative
        bucket.blob(blob_name).upload_from_filename(str(file_path))


def _resolve_adapter_path(adapter_path: str) -> Path:
    """
    Resolve adapter weights to a local directory.

    Local paths are returned as-is. GCS URIs are downloaded to a temp folder.
  """
    if _is_gcs_path(adapter_path):
        temp_dir = Path(tempfile.mkdtemp(prefix="code-sentinel-adapter-"))
        return _download_gcs_prefix(adapter_path, temp_dir)

    local = Path(adapter_path)
    if not local.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    return local


def _save_merged_model(local_output_dir: Path, output_path: str) -> None:
    """Persist merged weights locally or upload them to GCS."""
    if _is_gcs_path(output_path):
        _upload_directory_to_gcs(local_output_dir, output_path)
        print(f"Uploaded merged model to {output_path}")
        return

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in local_output_dir.iterdir():
        dest = output_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print(f"Saved merged model to {output_dir}")


def _push_to_huggingface_hub(local_model_dir: Path, repo_id: Optional[str] = None) -> None:
    """
    Push merged model and tokenizer to the Hugging Face Hub.

    Requires HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) and HF_REPO_ID env vars.
    """
    repo_id = repo_id or os.environ.get("HF_REPO_ID")
    if not repo_id:
        print("Skipping Hugging Face Hub upload (set HF_REPO_ID to enable).")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is required for Hub upload."
        )

    model = AutoModelForCausalLM.from_pretrained(local_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

    model.push_to_hub(repo_id, token=token)
    tokenizer.push_to_hub(repo_id, token=token)
    print(f"Pushed merged model to Hugging Face Hub: {repo_id}")


def merge_adapters(
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    *,
    hf_repo_id: Optional[str] = None,
) -> str:
    """
    Merge LoRA adapter weights into the base Mistral-7B model.

    Loads PEFT adapters, calls merge_and_unload(), then saves the merged
    checkpoint to GCS (or a local path) and optionally pushes to the Hub.

    Args:
        adapter_path: Local path or gs:// URI to PEFT adapter weights.
        output_path: Local path or gs:// URI for the merged model artifacts.
        hf_repo_id: Optional Hub repo id (e.g. username/code-sentinel-run1).

    Returns:
        The output_path where merged artifacts were written.
    """
    print(f"Loading base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base weights in fp16 on GPU when available; CPU otherwise.
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Resolve adapters (download from GCS when needed).
    local_adapter_dir = _resolve_adapter_path(adapter_path)
    print(f"Loading LoRA adapters from: {adapter_path}")

    peft_model = PeftModel.from_pretrained(base_model, str(local_adapter_dir))
    merged_model = peft_model.merge_and_unload()

    # Write merged weights to a staging directory before GCS/Hub upload.
    with tempfile.TemporaryDirectory(prefix="code-sentinel-merged-") as tmp:
        staging_dir = Path(tmp)
        print("Saving merged weights...")
        merged_model.save_pretrained(staging_dir)
        tokenizer.save_pretrained(staging_dir)

        _save_merged_model(staging_dir, output_path)
        _push_to_huggingface_hub(staging_dir, repo_id=hf_repo_id)

    print("Merge complete.")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge Code Sentinel LoRA adapters into Mistral-7B."
    )
    parser.add_argument(
        "--adapter-path",
        default=DEFAULT_ADAPTER_PATH,
        help=f"PEFT adapter directory (default: {DEFAULT_ADAPTER_PATH})",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Merged model destination (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Optional Hugging Face Hub repo id for upload.",
    )
    args = parser.parse_args()

    merge_adapters(
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        hf_repo_id=args.hf_repo_id,
    )
