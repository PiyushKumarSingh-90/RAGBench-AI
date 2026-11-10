"""RAGBench Streamlit dashboard (talks to the FastAPI backend over HTTP).

    uvicorn api.main:app                      # terminal 1
    streamlit run dashboard/app.py            # terminal 2 -> http://localhost:8501

Set RAGBENCH_API_URL if the API is not at http://127.0.0.1:8000.
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("RAGBENCH_API_URL", "http://127.0.0.1:8000")
STRATEGY_ORDER = ["dense_fixed", "dense_semantic", "hybrid", "rerank"]
COLORS = {"dense_fixed": "#2a78d6", "dense_semantic": "#eb6834", "hybrid": "#1baf7a",
          "rerank": "#eda100"}  # fmt: skip

st.set_page_config(page_title="RAGBench", layout="wide")


@st.cache_data(ttl=60)
def get(path: str):
    # Long timeout: a sleeping free-tier API can take about a minute to wake up.
    r = httpx.get(f"{API}{path}", timeout=120)
    r.raise_for_status()
    return r.json()


def post(path: str, payload: dict):
    try:
        r = httpx.post(f"{API}{path}", json=payload, timeout=300)
    except httpx.HTTPError as e:
        st.error(f"The API did not respond ({type(e).__name__}). It may be restarting; try again.")
        return None
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:  # e.g. an HTML error page from the hosting proxy
            detail = "the API is unavailable or restarting; try again in a minute"
        st.error(f"API error {r.status_code}: {detail}")
        return None
    return r.json()


try:
    strategies = get("/strategies")["strategies"]
    gens = get("/generators")["available"]
except httpx.HTTPError as e:
    st.error(
        f"Cannot reach the RAGBench API at {API} ({type(e).__name__}). If it is hosted on a "
        "free plan it may be waking up: wait a minute and reload. Locally, start it with "
        "`uvicorn api.main:app`."
    )
    st.stop()

st.title("RAGBench")
st.caption(
    "Compare four retrieval strategies over the same documentation corpus "
    "(Python, scikit-learn, Docker, FastAPI, AWS)."
)

with st.sidebar:
    st.header("Settings")
    generator_options = [g for g, ok in gens.items() if ok]
    generator = st.selectbox(
        "Answer generator",
        generator_options,
        help="LLM generators appear only when their API key is set where the API runs.",
    )
    missing = [g for g, ok in gens.items() if not ok]
    if missing:
        st.caption("Unavailable (no API key): " + ", ".join(missing))
    top_k = st.slider("Passages retrieved (top k)", 1, 10, 5)
    st.divider()
    st.subheader("Strategies")
    for name, desc in strategies.items():
        st.markdown(f"**{name}** — {desc}")


def show_result(res: dict, expanded_context: bool = False):
    if res["abstained"]:
        st.warning(res["answer"])
    else:
        st.markdown(res["answer"] or "_(empty answer)_")
    t = res["timings_ms"]
    cols = st.columns(3)
    cols[0].metric("Retrieval", f"{t.get('retrieval_total', 0):.0f} ms")
    cols[1].metric("Generation", f"{t.get('generation', 0):.0f} ms")
    cols[2].metric("Total", f"{t.get('total', 0):.0f} ms")
    if res["citations"]:
        st.markdown("**Sources cited**")
        for c in res["citations"]:
            st.markdown(f"- [{c['document_title']} — {c['section_heading']}]({c['source_url']})")
    cited = {c["chunk_id"] for c in res["citations"]}
    with st.expander(f"Retrieved context ({len(res['retrieved'])} passages)", expanded_context):
        for r in res["retrieved"]:
            mark = " · cited" if r["chunk_id"] in cited else ""
            st.markdown(
                f"**[{r['rank']}] {r['document_id']} — {r['section_heading']}** "
                f"(score {r['score']:.3f}){mark}"
            )
            st.text(r["text"][:1500])


ask, compare, evaluation, explore = st.tabs(
    ["Ask", "Compare strategies", "Evaluation results", "Benchmark explorer"]
)

with ask:
    strategy = st.radio(
        "Retrieval strategy", STRATEGY_ORDER, index=2, horizontal=True, key="ask_strategy"
    )
    q = st.text_input("Question", placeholder="How do I raise a 404 error in FastAPI?")
    if st.button("Ask", type="primary", disabled=not q):
        with st.spinner("Retrieving and answering…"):
            res = post(
                "/query",
                {
                    "question": q,
                    "strategy": strategy,
                    "generator": generator,
                    "top_k": top_k,
                },  # fmt: skip
            )
        if res:
            show_result(res, expanded_context=True)

with compare:
    st.caption("Runs the same question through all four strategies with the same generator.")
    cq = st.text_input("Question", key="cmp_q", placeholder="What does docker compose down do?")
    if st.button("Compare", type="primary", disabled=not cq):
        with st.spinner("Running all four strategies…"):
            res = post("/compare", {"question": cq, "generator": generator, "top_k": top_k})
        if res:
            rows = [
                {
                    "strategy": s,
                    "abstained": r["abstained"],
                    "top document": r["retrieved"][0]["document_id"] if r["retrieved"] else "",
                    "documents cited": ", ".join(
                        dict.fromkeys(c["document_id"] for c in r["citations"])
                    ),
                    "retrieval ms": round(r["timings_ms"]["retrieval_total"], 1),
                    "total ms": round(r["timings_ms"]["total"], 1),
                }
                for s, r in res.items()
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            for col, (s, r) in zip(st.columns(len(res)), res.items()):
                with col:
                    st.subheader(s)
                    show_result(r)

with evaluation:
    try:
        ev = get("/evaluation")
    except httpx.HTTPError:
        ev = None
    if not ev:
        st.info("No evaluation results yet. Run `python scripts/evaluate.py`.")
    else:
        meta, m = ev["meta"], ev["strategies"]
        st.caption(
            f"Saved benchmark run `{meta['run']}` · generator `{meta['generator']}` · "
            f"NLI judge `{meta['nli_model']}`. 50 questions (40 answerable, 10 unanswerable)."
        )
        names = [s for s in STRATEGY_ORDER if s in m]
        retrieval = pd.DataFrame(
            {k: [m[s][k] for s in names] for k in
             ["context_recall", "context_precision", "map", "hit_at_1", "hit_at_3", "hit_at_5"]},
            index=names,
        )  # fmt: skip
        answers = pd.DataFrame(
            {k: [m[s][k] for s in names] for k in
             ["faithfulness", "citation_support", "fact_coverage", "token_f1",
              "semantic_similarity"]},
            index=names,
        )  # fmt: skip
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Retrieval quality")
            st.bar_chart(retrieval.T, stack=False, color=[COLORS[s] for s in names])
            st.dataframe(retrieval.style.format("{:.3f}"), width="stretch")
        with c2:
            st.subheader("Answer quality")
            st.bar_chart(answers.T, stack=False, color=[COLORS[s] for s in names])
            st.dataframe(answers.style.format("{:.3f}"), width="stretch")
            st.caption(
                "Faithfulness and citation support come from a strict local NLI judge and are "
                "lower bounds (see docs/phase5_evaluation.md)."
            )
        st.subheader("Latency (median ms)")
        stages = ["embedding", "bm25", "rerank", "retrieval_total", "generation", "total"]
        lat = pd.DataFrame(
            {st_: [m[s]["latency_ms"].get(st_, {}).get("median") for s in names] for st_ in stages},
            index=names,
        )
        st.bar_chart(
            lat[["embedding", "bm25", "rerank"]].fillna(0), color=["#2a78d6", "#1baf7a", "#eda100"]
        )
        st.dataframe(lat, width="stretch")
        st.caption("Generation time includes free-tier rate-limit waits.")
        st.subheader("Unanswerable questions")
        st.dataframe(
            pd.DataFrame(
                {
                    "correctly declined": [
                        f"{m[s]['unanswerable_abstained']}/{m[s]['unanswerable']}" for s in names
                    ],
                    "answerable wrongly declined": [
                        f"{m[s]['answerable_abstained']}/{m[s]['answerable']}" for s in names
                    ],
                },
                index=names,
            ),
            width="stretch",
        )

with explore:
    questions = get("/benchmark/questions")
    qmap = {q["question_id"]: q for q in questions}
    c1, c2, c3 = st.columns(3)
    src = c1.selectbox("Source", ["all"] + sorted({q["source_group"] for q in questions}))
    qtype = c2.selectbox("Type", ["all"] + sorted({q["question_type"] for q in questions}))
    es = c3.selectbox("Strategy", STRATEGY_ORDER, index=2, key="explore_strategy")
    ids = [
        q["question_id"]
        for q in questions
        if (src == "all" or q["source_group"] == src)
        and (qtype == "all" or q["question_type"] == qtype)
    ]
    qid = st.selectbox("Question", ids, format_func=lambda i: f"{i} — {qmap[i]['question']}")
    if qid:
        q = qmap[qid]
        st.markdown(f"**Expected answer:** {q['expected_answer']}")
        st.caption(
            "Answerable from corpus" if q["answerable_from_corpus"] else "Unanswerable by design"
        )
        rows = {r["question_id"]: r for r in get(f"/benchmark/results/{es}")}
        r = rows.get(qid)
        if not r:
            st.info("No saved answer for this strategy.")
        else:
            st.markdown(f"**Saved answer ({r.get('generator', 'groq')}):**")
            if r["abstained"]:
                st.warning(r["answer"])
            else:
                st.markdown(r["answer"])
            e = r.get("evaluation") or {}
            if e:
                keys = ["context_recall", "context_precision", "hit_at_5", "faithfulness",
                        "citation_support", "fact_coverage"]  # fmt: skip
                cols = st.columns(len(keys))
                for col, k in zip(cols, keys):
                    v = e.get(k)
                    shown = "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
                    col.metric(k.replace("_", " "), shown)
                if e.get("unsupported_claims"):
                    with st.expander("Claims the NLI judge could not match to the context"):
                        for c in e["unsupported_claims"]:
                            st.markdown(f"- {c}")
            with st.expander("Evidence (gold)"):
                for ev_ in q["evidence"]:
                    st.markdown(f"**{ev_['document_title']} — {ev_['section_heading']}**")
                    st.text(ev_["supporting_excerpt"])
