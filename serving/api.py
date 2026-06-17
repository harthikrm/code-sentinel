# FastAPI service for Code Sentinel merged-model code review inference.

from __future__ import annotations

import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.cloud import storage
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

DEFAULT_MODEL_PATH = "gs://code-sentinel-2026-training/merged-model/run1"
LOCAL_MODEL_CACHE_DIR = "/tmp/code-sentinel-model"
MAX_NEW_TOKENS = 256

# Module-level handle populated during application startup.
_inference_pipeline: Optional[Any] = None
_model_load_error: Optional[str] = None
_model_load_lock = threading.Lock()
_model_status = "pending"


def _is_gcs_path(path: str) -> bool:
    """Return True if path is a Google Cloud Storage URI."""
    return path.startswith("gs://")


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split gs://bucket/prefix into (bucket_name, blob_prefix)."""
    without_scheme = uri[len("gs://") :]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix.rstrip("/")


def _download_gcs_model(gcs_uri: str, local_dir: str) -> str:
    """
    Download all model artifacts under a GCS prefix to a local directory.

    Hugging Face ``from_pretrained`` / ``pipeline`` cannot load directly from
    ``gs://`` URIs — they interpret the string as a Hub repo id.

    Args:
        gcs_uri: GCS prefix containing merged model weights.
        local_dir: Local directory to populate.

    Returns:
        Local path passed to Hugging Face loaders.
    """
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No objects found at {gcs_uri}")

    root = Path(local_dir)
    root.mkdir(parents=True, exist_ok=True)

    def _download_blob(blob: storage.Blob) -> None:
        if blob.name.endswith("/"):
            return
        relative = blob.name[len(prefix) + 1 :] if prefix else blob.name
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            print(f"Downloading gs://{bucket_name}/{blob.name} -> {destination}")
            blob.download_to_filename(str(destination))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_download_blob, blob) for blob in blobs]
        for future in as_completed(futures):
            future.result()

    return str(root)


def _resolve_model_path(model_path: str) -> str:
    """Return a local path suitable for Hugging Face model loading."""
    if _is_gcs_path(model_path):
        return _download_gcs_model(model_path, LOCAL_MODEL_CACHE_DIR)
    return model_path


def format_review_prompt(diff: str, lang: str) -> str:
    """
    Build the Mistral [INST] prompt used during training (see data/preprocess.py).

    The model is asked to review a diff; generation continues after [/INST].
    """
    return f"""[INST] Review the following code change and identify issues:
Language: {lang}
Diff: {diff}
Provide specific, actionable feedback.[/INST]
"""


class ReviewRequest(BaseModel):
    """Incoming code review request."""

    diff: str = Field(..., description="Unified diff or code change to review.")
    lang: str = Field(default="py", description="Programming language of the change.")


class ReviewResponse(BaseModel):
    """Generated review text."""

    review: str


class HealthResponse(BaseModel):
    """Health probe response."""

    status: str
    model: str
    detail: Optional[str] = None


class PredictRequest(BaseModel):
    """Vertex AI online prediction request envelope."""

    instances: List[ReviewRequest] = Field(
        ...,
        description="One or more review requests in Vertex instances format.",
    )


class PredictResponse(BaseModel):
    """Vertex AI online prediction response envelope."""

    predictions: List[ReviewResponse] = Field(
        ...,
        description="Generated reviews aligned with input instances.",
    )


def _load_generation_pipeline(model_path: str) -> Any:
    """
    Load a Hugging Face text-generation pipeline for the merged model.

    GCS URIs are downloaded to ``LOCAL_MODEL_CACHE_DIR`` first.
    """
    local_path = _resolve_model_path(model_path)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        return pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=dtype)
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=dtype,
        device=-1,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def _extract_review(generated_text: str, prompt: str) -> str:
    """Strip the prompt prefix and return only the model-generated review."""
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].strip()
    if "[/INST]" in generated_text:
        return generated_text.split("[/INST]", maxsplit=1)[-1].strip()
    return generated_text.strip()


def _load_model_in_background(model_path: str) -> None:
    """
    Load the inference pipeline in a background thread.

    Vertex health checks require ``/health`` to respond before model weights
    finish downloading from GCS. Loading in the lifespan hook blocks uvicorn
    from accepting requests and causes deploy timeouts.
    """
    global _inference_pipeline, _model_load_error, _model_status

    try:
        _model_status = "loading"
        print(f"Loading Code Sentinel model from: {model_path}")
        loaded_pipeline = _load_generation_pipeline(model_path)
        with _model_load_lock:
            _inference_pipeline = loaded_pipeline
        _model_status = "ready"
        print("Model loaded successfully.")
    except Exception as exc:
        _model_load_error = str(exc)
        _model_status = "error"
        print(f"Model load failed: {exc}")
        traceback.print_exc()


def _wait_for_model() -> Any:
    """
    Return the loaded pipeline if ready.

    Raises:
        RuntimeError: If loading failed or the model is not ready yet.
    """
    global _inference_pipeline, _model_load_error

    if _inference_pipeline is not None:
        return _inference_pipeline
    if _model_load_error:
        raise RuntimeError(f"Inference pipeline failed to load: {_model_load_error}")
    raise RuntimeError(
        "Model is still loading. Vertex routes traffic only after /health returns ready."
    )


def _generate_review(diff: str, lang: str) -> str:
    """
    Run the loaded text-generation pipeline for a single diff.

    Args:
        diff: Unified diff or code change to review.
        lang: Programming language tag (e.g. ``py``).

    Returns:
        Generated review comment text.

    Raises:
        RuntimeError: If the inference pipeline has not been initialized.
    """
    pipe = _wait_for_model()
    prompt = format_review_prompt(diff=diff, lang=lang)
    outputs = pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=float(os.environ.get("GENERATION_TEMPERATURE", "0.2")),
        top_p=float(os.environ.get("GENERATION_TOP_P", "0.9")),
        return_full_text=True,
    )
    generated = outputs[0]["generated_text"]
    return _extract_review(generated, prompt)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start background model loading and release resources on shutdown.

    ``/health`` returns 503 until the model is ready so Vertex does not route
    predict traffic while weights are still downloading from GCS.
    """
    global _inference_pipeline, _model_load_error

    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    loader = threading.Thread(
        target=_load_model_in_background,
        args=(model_path,),
        daemon=True,
        name="model-loader",
    )
    loader.start()

    yield

    _inference_pipeline = None
    _model_load_error = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model resources released.")


app = FastAPI(
    title="Code Sentinel API",
    description="QLoRA fine-tuned Mistral-7B code review service.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow browser clients (React frontend) to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """
    Liveness probe for Vertex.

    Returns 200 only when the model is loaded and ready for inference.
    Returns 503 while weights download/load so predict is not routed early.
    """
    if _model_status == "ready":
        return {"status": "ok", "model": "ready"}
    if _model_status == "error":
        return JSONResponse(
            status_code=500,
            content={"status": "error", "model": "error", "detail": _model_load_error},
        )
    return JSONResponse(
        status_code=503,
        content={
            "status": "loading",
            "model": _model_status,
            "detail": "Downloading and loading model weights from GCS",
        },
    )


@app.post("/review", response_model=ReviewResponse)
async def review(request: ReviewRequest) -> ReviewResponse:
    """
    Generate a code review comment for the supplied diff.

    Formats input with the training [INST] template, runs text generation,
    and returns only the newly generated review text.
    """
    try:
        review_text = _generate_review(diff=request.diff, lang=request.lang)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ReviewResponse(review=review_text)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Vertex AI online prediction route.

    Accepts the standard ``instances`` / ``predictions`` JSON envelope used by
    Vertex custom containers (see ``deploy_api.py``).
    """
    predictions = []
    for item in request.instances:
        try:
            review_text = _generate_review(diff=item.diff, lang=item.lang)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        predictions.append(ReviewResponse(review=review_text))
    return PredictResponse(predictions=predictions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        reload=False,
    )
