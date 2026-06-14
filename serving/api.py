# FastAPI service for Code Sentinel merged-model code review inference.

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline

DEFAULT_MODEL_PATH = "gs://code-sentinel-training/merged-model/run1"
MAX_NEW_TOKENS = 256

# Module-level handle populated during application startup.
_inference_pipeline: Optional[Any] = None


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

    Supports local paths and gs:// URIs (requires gcsfs / GCS credentials).
    """
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    return pipeline(
        task="text-generation",
        model=model_path,
        tokenizer=model_path,
        torch_dtype=dtype,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def _extract_review(generated_text: str, prompt: str) -> str:
    """Strip the prompt prefix and return only the model-generated review."""
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].strip()
    if "[/INST]" in generated_text:
        return generated_text.split("[/INST]", maxsplit=1)[-1].strip()
    return generated_text.strip()


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
    if _inference_pipeline is None:
        raise RuntimeError("Inference pipeline is not initialized.")

    prompt = format_review_prompt(diff=diff, lang=lang)
    outputs = _inference_pipeline(
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
    Load the merged model on startup and release resources on shutdown.
    """
    global _inference_pipeline

    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    print(f"Loading Code Sentinel model from: {model_path}")
    _inference_pipeline = _load_generation_pipeline(model_path)
    print("Model loaded successfully.")

    yield

    _inference_pipeline = None
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


@app.get("/health", response_model=HealthResponse)
async def health() -> Dict[str, str]:
    """Liveness probe for load balancers and Vertex health checks."""
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
async def review(request: ReviewRequest) -> ReviewResponse:
    """
    Generate a code review comment for the supplied diff.

    Formats input with the training [INST] template, runs text generation,
    and returns only the newly generated review text.
    """
    review_text = _generate_review(diff=request.diff, lang=request.lang)
    return ReviewResponse(review=review_text)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Vertex AI online prediction route.

    Accepts the standard ``instances`` / ``predictions`` JSON envelope used by
    Vertex custom containers (see ``deploy_api.py``).
    """
    predictions = [
        ReviewResponse(review=_generate_review(diff=item.diff, lang=item.lang))
        for item in request.instances
    ]
    return PredictResponse(predictions=predictions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        reload=False,
    )
