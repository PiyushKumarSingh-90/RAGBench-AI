"""Phase 5 evaluation metrics (offline, no API calls).

Retrieval metrics use the verbatim evidence excerpts stored with each benchmark
question. Answer metrics use local models: a cross-encoder NLI model for
faithfulness, citation support and fact coverage, and the MiniLM embedder for
semantic similarity. Every metric is a measured proxy; see
docs/phase5_evaluation.md for definitions and known limitations.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

CITE_RE = re.compile(r"[\[【](\d+)(?:†[^\]】]*)?[\]】]")
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NGRAM = 3
EVIDENCE_COVERED = 0.8  # share of an excerpt's word 3-grams that must be retrieved
CHUNK_RELEVANT = 0.5  # share of an excerpt's 3-grams a single chunk must contain
ENTAIL_THRESHOLD = 0.5


# --- text helpers ------------------------------------------------------------


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    w = words(text)
    if len(w) < n:
        return {tuple(w)} if w else set()
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def coverage(excerpt: str, text_ngrams: set) -> float:
    g = ngrams(excerpt)
    return len(g & text_ngrams) / len(g) if g else 0.0


def strip_markup(text: str) -> str:
    text = CITE_RE.sub("", text)
    text = re.sub(r"[*_`#>|]+", " ", text)
    text = re.sub(r"^\s*([-+]|\d+\.)\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Claim:
    text: str  # markup and citation markers removed
    cited: list[int]  # 1-based passage numbers cited in this claim


def split_claims(answer: str, min_words: int = 4) -> list[Claim]:
    """Split an answer into sentence-level claims with their citation markers.

    A segment that is only citation markers (e.g. "[2]" after the full stop) is
    attached to the previous claim. Fenced code blocks and very short fragments
    (headings, table separators) are dropped.
    """
    segments = []
    answer = re.sub(r"```.*?(```|$)", "\n", answer, flags=re.S)  # code blocks are not claims
    for line in answer.splitlines():
        if re.fullmatch(r"\s*\|?[\s:|-]*\|?\s*", line):  # blank or table rule
            continue
        segments += re.split(r"(?<=[.!?])\s+", line)
    claims: list[Claim] = []
    for seg in segments:
        cited = [int(n) for n in CITE_RE.findall(seg)]
        text = strip_markup(seg)
        if len(words(text)) < min_words:
            if cited and claims:
                claims[-1].cited += [c for c in cited if c not in claims[-1].cited]
            continue
        claims.append(Claim(text, list(dict.fromkeys(cited))))
    return claims


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 (lower-case, punctuation and articles removed)."""

    def norm(s: str) -> list[str]:
        s = strip_markup(s).lower()
        s = "".join(ch if ch not in string.punctuation else " " for ch in s)
        return [t for t in s.split() if t not in {"a", "an", "the"}]

    p, r = norm(prediction), norm(reference)
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


# --- retrieval metrics -------------------------------------------------------


def retrieval_metrics(question: dict, retrieved: list[dict], chunk_text: dict[str, str]) -> dict:
    """Context recall/precision and hit@k for one answerable question.

    `retrieved` is the run's ranked list (dicts with chunk_id, document_id,
    headings_covered). A chunk is relevant if it contains at least half of one
    evidence excerpt's word 3-grams. Context recall is the mean, over evidence
    excerpts, of the share of the excerpt's 3-grams present anywhere in the top-k.
    """
    excerpts = [e["supporting_excerpt"] for e in question["evidence"]]
    grams = [ngrams(chunk_text[r["chunk_id"]]) for r in retrieved]
    union = set().union(*grams) if grams else set()
    per_excerpt = [coverage(x, union) for x in excerpts]
    relevant = [any(coverage(x, g) >= CHUNK_RELEVANT for x in excerpts) for g in grams]
    k = len(retrieved)
    first = next((i + 1 for i, rel in enumerate(relevant) if rel), None)
    hits_at_rel = [sum(relevant[: i + 1]) / (i + 1) for i, rel in enumerate(relevant) if rel]
    return {
        "context_recall": sum(per_excerpt) / len(per_excerpt),
        "evidence_fully_retrieved": sum(c >= EVIDENCE_COVERED for c in per_excerpt)
        / len(per_excerpt),
        "context_precision": sum(relevant) / k if k else 0.0,
        "average_precision": sum(hits_at_rel) / len(hits_at_rel) if hits_at_rel else 0.0,
        "hit_at_1": first is not None and first <= 1,
        "hit_at_3": first is not None and first <= 3,
        "hit_at_5": first is not None and first <= 5,
        "relevant_ranks": [i + 1 for i, rel in enumerate(relevant) if rel],
    }


# --- answer metrics ----------------------------------------------------------


class NLIScorer:
    """Entailment probabilities from a local cross-encoder NLI model."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, max_length=512)
        self.model = model
        self.name = model_name
        labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        self.entail_idx = labels["entailment"]

    def entail(self, pairs: list[tuple[str, str]]) -> list[float]:
        """P(entailment) for each (premise, hypothesis) pair."""
        if not pairs:
            return []
        probs = self.model.predict(pairs, apply_softmax=True, batch_size=16)
        return [float(p[self.entail_idx]) for p in probs]


def answer_metrics(
    answer: str,
    expected: str,
    contexts: list[str],
    nli: NLIScorer,
    similarity: float | None = None,
) -> dict:
    """Faithfulness, citation support and correctness proxies for one answer.

    - faithfulness: share of answer claims entailed by at least one retrieved passage.
    - citation_support: share of cited claims entailed by at least one cited passage;
      citation_precision: share of individual citations whose passage entails the claim.
    - fact_coverage: share of expected-answer sentences entailed by the answer.
    """
    claims = split_claims(answer)
    pairs, index = [], []
    for ci, c in enumerate(claims):
        for pi, ctx in enumerate(contexts):
            pairs.append((ctx, c.text))
            index.append((ci, pi + 1))
    exp_sents = [s.text for s in split_claims(expected, min_words=3)] or [strip_markup(expected)]
    clean_answer = strip_markup(answer)
    probs = nli.entail(pairs + [(clean_answer, s) for s in exp_sents])
    claim_probs, fact_probs = probs[: len(pairs)], probs[len(pairs) :]

    table: dict[tuple[int, int], float] = dict(zip(index, claim_probs))
    supported = [
        max((table[(ci, p)] for p in range(1, len(contexts) + 1)), default=0.0) >= ENTAIL_THRESHOLD
        for ci in range(len(claims))
    ]
    cited = [(ci, c) for ci, c in enumerate(claims) if c.cited]
    cite_ok = [
        max(table.get((ci, p), 0.0) for p in c.cited) >= ENTAIL_THRESHOLD for ci, c in cited
    ]
    single = [table.get((ci, p), 0.0) >= ENTAIL_THRESHOLD for ci, c in cited for p in c.cited]
    out = {
        "claims": len(claims),
        "faithfulness": sum(supported) / len(supported) if supported else None,
        "unsupported_claims": [claims[i].text for i, ok in enumerate(supported) if not ok],
        "cited_claims": len(cited),
        "citation_support": sum(cite_ok) / len(cite_ok) if cite_ok else None,
        "citation_precision": sum(single) / len(single) if single else None,
        "fact_coverage": sum(p >= ENTAIL_THRESHOLD for p in fact_probs) / len(fact_probs),
        "token_f1": token_f1(answer, expected),
    }
    if similarity is not None:
        out["semantic_similarity"] = similarity
    return out
