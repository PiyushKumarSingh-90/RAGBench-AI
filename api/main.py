"""RAGBench FastAPI backend.

    uvicorn api.main:app --reload            # http://127.0.0.1:8000/docs

LLM generators are enabled only when their key (e.g. GROQ_API_KEY) is set in
this process's environment. The offline extractive generator always works.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ragbench.service import STRATEGIES, RAGService, StrategyDisabled, disabled_strategies

Strategy = Literal["dense_fixed", "dense_semantic", "hybrid", "rerank"]
GeneratorName = Literal["extractive", "groq", "gemini", "claude"]

app = FastAPI(title="RAGBench API", version="0.6.0")
service = RAGService()


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    strategy: Strategy = "hybrid"
    generator: GeneratorName = "extractive"
    top_k: int = Field(5, ge=1, le=20)


class CompareRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    generator: GeneratorName = "extractive"
    strategies: list[Strategy] = list(STRATEGIES)
    top_k: int = Field(5, ge=1, le=20)


def _run(question: str, strategy: str, generator: str, top_k: int) -> dict:
    try:
        return service.query(question, strategy, generator, top_k)
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except StrategyDisabled as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503, detail=f"Index missing; run scripts/build_index.py ({e})"
        ) from e
    except Exception as e:  # upstream LLM errors (rate limit, auth) surface as 502
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/strategies")
def strategies() -> dict:
    return {
        "strategies": STRATEGIES,
        "default": "hybrid",
        "disabled": sorted(disabled_strategies()),
    }


@app.get("/generators")
def generators() -> dict:
    return {"available": service.available_generators(), "default": "extractive"}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    return _run(req.question, req.strategy, req.generator, req.top_k)


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    off = disabled_strategies()
    return {
        s: _run(req.question, s, req.generator, req.top_k) for s in req.strategies if s not in off
    }


@app.get("/benchmark/questions")
def benchmark_questions() -> list[dict]:
    return service.benchmark_questions()


@app.get("/benchmark/results/{strategy}")
def benchmark_results(strategy: Strategy) -> list[dict]:
    return service.benchmark_results(strategy)


@app.get("/evaluation")
def evaluation() -> dict:
    metrics = service.evaluation_metrics()
    if metrics is None:
        raise HTTPException(status_code=404, detail="Run scripts/evaluate.py first")
    return metrics
