# RAGBench API + dashboard image (CPU only). Untested: Docker was not available
# on the development machine. See docs/phase8_deployment.md.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

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
RUN python scripts/build_index.py --chunking fixed \
 && python scripts/build_index.py --chunking semantic \
 && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
