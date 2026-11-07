# Phase 6 — Application (FastAPI backend + Streamlit dashboard)

Status: complete, awaiting approval (2026-11-07).

## Run it

```bash
# terminal 1: backend (http://127.0.0.1:8000, interactive docs at /docs)
.venv/bin/uvicorn api.main:app

# terminal 2: dashboard (http://localhost:8501)
.venv/bin/streamlit run dashboard/app.py
```

Both need the indexes from Phase 3/4 (`scripts/build_index.py --chunking fixed|semantic`).
The dashboard reads `RAGBENCH_API_URL` (default `http://127.0.0.1:8000`).

To answer with an LLM, export its key **in the terminal that runs the API**
(e.g. `export GROQ_API_KEY=...`). Without a key only the offline extractive
generator is offered. Keys are never sent to or stored by the dashboard.

## Architecture

| Layer | File | Role |
|---|---|---|
| Service | `src/ragbench/service.py` | Loads indexes, retrievers and generators once (lazily, thread-safe); runs queries; reads saved benchmark and Phase 5 results |
| API | `api/main.py` | FastAPI endpoints with request validation |
| Dashboard | `dashboard/app.py` | Streamlit UI; talks to the API over HTTP only |

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/strategies` | The four strategies with one-line descriptions |
| GET | `/generators` | Which generators are usable (key present) |
| POST | `/query` | `{question, strategy, generator, top_k}` → answer, abstention, citations, retrieved passages with text, timings |
| POST | `/compare` | Same question through several strategies |
| GET | `/benchmark/questions` | The 50 benchmark questions |
| GET | `/benchmark/results/{strategy}` | Saved Groq answers merged with their Phase 5 metrics |
| GET | `/evaluation` | Phase 5 aggregate metrics |

Errors: 400 if an LLM generator is chosen without its key; 422 for invalid input;
502 for upstream LLM errors (e.g. Groq rate limit); 503 if indexes are missing.

## Dashboard

- **Ask** — pick a strategy, ask a question; shows the answer, cited sources (linked),
  retrieval/generation/total latency and the retrieved passages, with cited ones marked.
- **Compare strategies** — one question through all four strategies side by side,
  with a summary table (top document, documents cited, latency).
- **Evaluation results** — Phase 5 retrieval and answer metrics, latency by stage and
  unanswerable-question handling, as charts and tables.
- **Benchmark explorer** — filter the 50 questions by source and type; see the expected
  answer, the gold evidence, the saved Groq answer per strategy, its Phase 5 scores and
  the claims the NLI judge could not match.

The sidebar describes each strategy, so the differences are visible without reading code.

## Verification (2026-11-07)

- `tests/test_api.py` (4 tests, real indexes, extractive generator): metadata endpoints,
  `/query` shape, 400 without a key, 422 on bad input, saved results and evaluation.
  Full suite: 61 passed; Ruff clean.
- Live check with `uvicorn`: `/compare` for "How do I raise a 404 error in FastAPI?"
  returned `fastapi_tutorial_handling_errors` as the top document for all four strategies.
- Dashboard exercised headlessly with Streamlit's `AppTest` against the running API:
  all four tabs render; Ask and Compare complete with no exceptions or errors.
  The in-app browser could not open the page in this session, so the layout was not
  checked visually.
- Versions: FastAPI 0.162.0, Streamlit 1.67.0.

## Limitations

- No authentication or rate limiting: intended for local use.
- Models load on the first request for each strategy (first `rerank` call is slower).
- LLM calls are synchronous; a Groq rate-limit wait blocks that request.
