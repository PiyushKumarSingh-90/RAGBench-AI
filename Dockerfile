# RAGBench API image (CPU only). Used by docker-compose and by Render (render.yaml).
# Not built on the development machine (no Docker there). See docs/render_deployment.md.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY api ./api
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY data/processed ./data/processed
COPY evaluation ./evaluation
COPY results ./results

# Build both indexes at image build time (downloads MiniLM once into the image)
# and pre-fetch the reranker so the first rerank query does not download it.
RUN HF_HUB_OFFLINE=0 python scripts/build_index.py --chunking fixed \
 && HF_HUB_OFFLINE=0 python scripts/build_index.py --chunking semantic \
 && HF_HUB_OFFLINE=0 python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"

EXPOSE 8000 8501
# Render (and similar hosts) pass the port in $PORT; default 8000 for docker compose.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
