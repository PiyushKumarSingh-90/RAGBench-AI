"""ClaudeGenerator tests with a fake Anthropic client (no network, no key)."""

from types import SimpleNamespace

from ragbench.chunking import Chunk
from ragbench.generation import ABSTAIN_TEXT, SYSTEM_PROMPT, build_prompt
from ragbench.llm import DEFAULT_MODEL, ClaudeGenerator
from ragbench.vectorstore import SearchResult


def ctx(n=2):
    return [
        SearchResult(
            Chunk(f"c{i}", f"d{i}", "src", f"T{i}", f"https://x/{i}", f"H{i}", 0, f"text {i}"),
            0.5,
            i,
        )
        for i in range(1, n + 1)
    ]


class FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = [SimpleNamespace(type="thinking", thinking="")]
        if self.text is not None:
            content.append(SimpleNamespace(type="text", text=self.text))
        return SimpleNamespace(
            content=content,
            model=kwargs["model"],
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category="cyber")
            if self.stop_reason == "refusal"
            else None,
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            _request_id="req_test",
        )


def gen(text, stop_reason="end_turn"):
    messages = FakeMessages(text, stop_reason)
    return ClaudeGenerator(client=SimpleNamespace(messages=messages)), messages


def test_request_shape_and_citation_mapping():
    g, m = gen("Volumes persist data [2]. Also [1][9].")
    ans = g.generate("How?", ctx(), confidence=0.01)  # confidence must not cause abstention
    req = m.calls[0]
    assert req["model"] == DEFAULT_MODEL == "claude-haiku-5-5"
    assert req["system"] == SYSTEM_PROMPT
    assert req["output_config"] == {"effort": "medium"}
    assert "thinking" not in req and "temperature" not in req
    user = req["messages"][0]
    assert (
        user["role"] == "user"
        and "[1] T1 — H1" in user["content"]
        and "Question: How?" in user["content"]
    )
    assert not ans.abstained
    assert ans.cited_chunk_ids == ["c2", "c1"]  # out-of-range [9] ignored
    assert ans.meta["stop_reason"] == "end_turn" and ans.meta["input_tokens"] == 100


def test_model_abstention_is_detected():
    g, _ = gen(ABSTAIN_TEXT)
    ans = g.generate("Unknown?", ctx())
    assert ans.abstained and ans.text == ABSTAIN_TEXT and ans.cited_chunk_ids == []


def test_refusal_is_recorded_not_raised():
    g, _ = gen(None, stop_reason="refusal")
    ans = g.generate("q?", ctx())
    assert ans.text == "" and not ans.abstained
    assert ans.meta["stop_reason"] == "refusal" and ans.meta["refusal_category"] == "cyber"


def test_build_prompt_contains_system_and_context():
    p = build_prompt("q?", ctx())
    assert p.startswith(SYSTEM_PROMPT) and ABSTAIN_TEXT in p and "[2] T2" in p


# --- Gemini ------------------------------------------------------------------

from ragbench.llm import GEMINI_DEFAULT_MODEL, GeminiGenerator  # noqa: E402


class FakeModels:
    def __init__(self, text, finish="STOP", block=None):
        self.text, self.finish, self.block, self.calls = text, finish, block, []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        parts = [SimpleNamespace(text=self.text)] if self.text is not None else []
        cand = SimpleNamespace(
            content=SimpleNamespace(parts=parts),
            finish_reason=SimpleNamespace(name=self.finish),
        )
        return SimpleNamespace(
            candidates=[cand],
            prompt_feedback=SimpleNamespace(block_reason=self.block) if self.block else None,
            usage_metadata=SimpleNamespace(prompt_token_count=90, candidates_token_count=15),
            model_version="gemini-2.0-flash-001",
            response_id="resp_1",
        )


def gem(text, **kw):
    models = FakeModels(text, **kw)
    return GeminiGenerator(client=SimpleNamespace(models=models)), models


def test_gemini_request_and_citations():
    g, m = gem("Answer [1] and [2].")
    ans = g.generate("How?", ctx(), confidence=0.0)
    req = m.calls[0]
    assert req["model"] == GEMINI_DEFAULT_MODEL == "gemini-flash-latest"
    assert req["config"].system_instruction == SYSTEM_PROMPT
    assert req["config"].temperature == 0.0
    assert "Question: How?" in req["contents"]
    assert ans.cited_chunk_ids == ["c1", "c2"] and not ans.abstained
    assert ans.meta["model"] == "gemini-2.0-flash-001" and ans.meta["input_tokens"] == 90


def test_gemini_abstention_and_block():
    g, _ = gem(ABSTAIN_TEXT)
    assert g.generate("q?", ctx()).abstained
    g, _ = gem(None, finish="SAFETY")
    ans = g.generate("q?", ctx())
    assert ans.text == "" and ans.meta["stop_reason"] == "refusal"


class Flaky(FakeModels):
    """Fails with the given codes first, then succeeds."""

    def __init__(self, codes):
        super().__init__("ok [1]")
        self.codes = list(codes)

    def generate_content(self, **kwargs):
        if self.codes:
            err = Exception("boom")
            err.code = self.codes.pop(0)
            raise err
        return super().generate_content(**kwargs)


def test_gemini_retries_transient_errors_only():
    import pytest

    models = Flaky([503, 429])
    g = GeminiGenerator(client=SimpleNamespace(models=models), base_delay=0)
    assert g.generate("q?", ctx()).cited_chunk_ids == ["c1"]
    assert len(models.calls) == 1  # succeeded on the third attempt

    g = GeminiGenerator(client=SimpleNamespace(models=Flaky([404])), base_delay=0)
    with pytest.raises(Exception, match="boom"):
        g.generate("q?", ctx())

    g = GeminiGenerator(
        client=SimpleNamespace(models=Flaky([503] * 3)), max_attempts=3, base_delay=0
    )
    with pytest.raises(Exception, match="boom"):
        g.generate("q?", ctx())


# --- Groq --------------------------------------------------------------------

from ragbench.llm import GROQ_DEFAULT_MODEL, GroqGenerator  # noqa: E402


class FakeCompletions:
    def __init__(self, text, finish="stop", fail_codes=()):
        self.text, self.finish, self.fail_codes, self.calls = text, finish, list(fail_codes), []

    def create(self, **kwargs):
        if self.fail_codes:
            err = Exception("groq boom")
            err.status_code = self.fail_codes.pop(0)
            raise err
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="chatcmpl_1",
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish, message=SimpleNamespace(content=self.text)
                )
            ],
            usage=SimpleNamespace(prompt_tokens=80, completion_tokens=12),
        )


def groq_gen(text, **kw):
    comp = FakeCompletions(text, **kw)
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    return GroqGenerator(client=client, base_delay=0), comp


def test_groq_request_citations_and_meta():
    g, c = groq_gen("Answer [2].")
    ans = g.generate("How?", ctx(), confidence=0.0)
    req = c.calls[0]
    assert req["model"] == GROQ_DEFAULT_MODEL and req["temperature"] == 0
    assert req["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Question: How?" in req["messages"][1]["content"]
    assert ans.cited_chunk_ids == ["c2"]
    assert ans.meta["input_tokens"] == 80 and ans.meta["stop_reason"] == "stop"


def test_groq_abstain_filter_and_retry():
    import pytest

    g, _ = groq_gen(ABSTAIN_TEXT)
    assert g.generate("q?", ctx()).abstained
    g, _ = groq_gen("", finish="content_filter")
    assert g.generate("q?", ctx()).meta["stop_reason"] == "refusal"
    g, _ = groq_gen("", finish="length")
    ans = g.generate("q?", ctx())
    assert ans.meta["stop_reason"] == "length" and ans.meta["empty_answer"] and not ans.abstained
    g, c = groq_gen("ok [1]", fail_codes=[429, 503])
    assert g.generate("q?", ctx()).cited_chunk_ids == ["c1"] and len(c.calls) == 1
    g, _ = groq_gen("ok", fail_codes=[401])
    with pytest.raises(Exception, match="groq boom"):
        g.generate("q?", ctx())
