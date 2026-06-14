# Container image for the Code Sentinel FastAPI inference service on Vertex AI.
#
# Build from the repository root:
#   docker build -f serving/Dockerfile -t us-central1-docker.pkg.dev/code-sentinel-499017/code-sentinel/api:latest .

FROM python:3.10-slim

WORKDIR /app

# Compilers needed for some pip wheels (e.g. sentencepiece).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/api.py serving/api.py
COPY data/utils.py data/utils.py

# Package directories required for `uvicorn serving.api:app`.
RUN touch serving/__init__.py data/__init__.py

EXPOSE 8080

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8080"]
