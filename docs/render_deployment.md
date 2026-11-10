# Deploying RAGBench on Render (free plan)

Two free web services, defined in `render.yaml`:

| Service | What runs | Built from |
|---|---|---|
| `ragbench-api` | FastAPI + MiniLM embeddings + cross-encoder reranker | `Dockerfile` (indexes built at image build time) |
| `ragbench-dashboard` | Streamlit only (no PyTorch) | `dashboard/requirements.txt` |

## Steps

1. Push the repository to GitHub (already done: `PiyushKumarSingh-90/RAGBench-AI`).
2. In Render: **New → Blueprint**, connect GitHub, choose `RAGBench-AI`. Render reads `render.yaml` and shows both services.
3. It asks for the two secret values:
   - `GROQ_API_KEY` (on `ragbench-api`): your Groq key. Leave it empty to run offline only.
   - `RAGBENCH_API_URL` (on `ragbench-dashboard`): enter `https://ragbench-api.onrender.com`. If Render gave the API a different URL (shown on the API service page), use that instead and redeploy the dashboard.
4. Click **Apply**. The API build takes about 10–15 minutes (PyTorch and models); the dashboard takes 2–3 minutes.
5. Check `https://<api-url>/health` → `{"status":"ok"}`, then open the dashboard URL.

Keys live only in Render's environment settings, never in the repository.

## Live deployment (2026-11-10)

The Blueprint sync failed without an error message, so the services were created by hand
(**New → Web Service**) with the same settings as `render.yaml`.

- Dashboard: `https://ragbench-dashboard-c1ab.onrender.com` (Python 3, free, Singapore; start
  command `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0
  --server.headless true`). The first deploy timed out because the start command was not set.
  Checked in a browser: the page loads, Groq is offered, and a live `hybrid` question returned an
  answer (about a minute on the first request while the API woke up).
- API: `https://ragbench-api-83p9.onrender.com` (Docker, free, Singapore). The first build
  failed because the installed package looked for data next to itself; fixed with
  `RAGBENCH_ROOT=/app` in the Dockerfile.
- Live checks: `/health` 200; `/generators` shows extractive and Groq available;
  `dense_fixed` and `hybrid` answered correctly (top document
  `fastapi_tutorial_handling_errors`). The first query took about 43 s (model load on
  0.1 CPU); later ones about 1 s.
- **`rerank` crashed the free instance** (HTTP 502, then the next query took 57 s because the
  service had restarted): on Render's Linux instance the cross-encoder does not fit in 512 MB
  next to the embedding model, although it measured 440 MB on macOS. It is therefore turned off
  with `RAGBENCH_DISABLED_STRATEGIES=rerank`: the API answers 503 with an explanation, `/compare`
  skips it, and the dashboard hides it and points to its saved benchmark results. Remove the
  variable on a plan with 2 GB to enable it.

## What to expect on the free plan

- **Sleep:** free services sleep after 15 minutes without traffic. The first visit then takes about a minute while both wake up; the dashboard shows "Cannot reach the RAGBench API… may be waking up" — wait and reload.
- **Memory:** the free plan has 512 MB. Three strategies fit; `rerank` does not (see above).
- **First query per strategy** loads its model, so it is slower (a few seconds); later queries are fast.
- **Groq:** with a key set, anyone using the public dashboard can choose Groq and uses your daily free tokens (200k/day). Remove the key in Render to turn this off.

## Verification before deployment (local, 2026-11-10)

- API started exactly as in the Dockerfile (`sh -c "uvicorn … --port ${PORT:-8000} --workers 1"`, `HF_HUB_OFFLINE=1`, one thread): `/health` OK.
- Dashboard installed in a fresh environment from `dashboard/requirements.txt` only (no PyTorch) and run against that API with Streamlit's `AppTest`: all four strategies answered with no errors; with the API stopped it shows the "waking up" message instead of crashing.
- `render.yaml` parses as valid YAML. Lint clean; 64 tests pass.

**Not verified:** the Docker image has not been built (no Docker on the development machine) and nothing has been deployed to Render yet. The first Render build is the first real build of the image.
