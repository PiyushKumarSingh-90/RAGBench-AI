"""LLM answer generators: Claude (Anthropic), Gemini (Google) and Groq.

Credentials are resolved by each SDK from the environment (Claude:
ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`; Gemini:
GEMINI_API_KEY or GOOGLE_API_KEY; Groq: GROQ_API_KEY). No key is stored in this repository.
"""

from __future__ import annotations

from ragbench.generation import (
    SYSTEM_PROMPT,
    GeneratedAnswer,
    build_user_message,
    parse_answer,
)
from ragbench.vectorstore import SearchResult

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_EFFORT = "medium"


class ClaudeGenerator:
    """Grounded answers from Claude. Abstention is decided by the model, not by the
    retrieval score (the `confidence` argument is ignored)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = 16000,
        client=None,
    ):
        if client is None:
            import anthropic  # lazy: only needed when Claude is actually used

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"claude(model={model},effort={effort})"

    def generate(
        self, question: str, contexts: list[SearchResult], confidence: float | None = None
    ) -> GeneratedAnswer:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": build_user_message(question, contexts)}],
        )
        usage = response.usage
        meta = {
            "model": response.model,
            "stop_reason": response.stop_reason,
            "request_id": getattr(response, "_request_id", None),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
        if response.stop_reason == "refusal":
            details = response.stop_details
            meta["refusal_category"] = getattr(details, "category", None) if details else None
            return GeneratedAnswer(text="", meta=meta)

        text = "".join(b.text for b in response.content if b.type == "text")
        answer = parse_answer(text, contexts)
        answer.meta = meta
        return answer


GEMINI_DEFAULT_MODEL = "gemini-flash-latest"


class GeminiGenerator:
    """Grounded answers from Google Gemini (google-genai SDK).

    The API key is read by the SDK from the GEMINI_API_KEY (or GOOGLE_API_KEY)
    environment variable; nothing is stored in the repo. Same prompt contract as
    ClaudeGenerator; abstention is decided by the model. Temperature 0 for
    repeatable benchmark runs.
    """

    def __init__(
        self,
        model: str = GEMINI_DEFAULT_MODEL,
        max_output_tokens: int = 2048,
        client=None,
        max_attempts: int = 6,
        base_delay: float = 2.0,
    ):
        if client is None:
            from google import genai  # lazy: only needed when Gemini is actually used

            client = genai.Client()
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.name = f"gemini(model={model},temperature=0)"

    def _call_with_retry(self, **kwargs):
        """Retry transient errors (429 rate limit, 5xx overload) with exponential backoff.
        Other errors (bad key, missing model, billing) are raised immediately."""
        import random
        import time

        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.models.generate_content(**kwargs)
            except Exception as e:
                code = getattr(e, "code", None)
                transient = code == 429 or (isinstance(code, int) and code >= 500)
                if not transient or attempt == self.max_attempts:
                    raise
                delay = min(self.base_delay * 2 ** (attempt - 1), 60) + random.uniform(0, 1)
                print(f"  Gemini {code}; retry {attempt}/{self.max_attempts - 1} in {delay:.0f}s")
                time.sleep(delay)

    def generate(
        self, question: str, contexts: list[SearchResult], confidence: float | None = None
    ) -> GeneratedAnswer:
        from google.genai import types

        response = self._call_with_retry(
            model=self.model,
            contents=build_user_message(question, contexts),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        usage = response.usage_metadata
        candidate = response.candidates[0] if response.candidates else None
        finish = getattr(candidate, "finish_reason", None)
        block = getattr(response.prompt_feedback, "block_reason", None)
        meta = {
            "model": getattr(response, "model_version", None) or self.model,
            "stop_reason": str(getattr(finish, "name", finish)) if finish else None,
            "request_id": getattr(response, "response_id", None),
            "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
            "output_tokens": (usage.candidates_token_count or 0) if usage else 0,
        }
        text = ""
        if candidate and candidate.content and candidate.content.parts:
            text = "".join(p.text or "" for p in candidate.content.parts)
        if (
            block
            or meta["stop_reason"] in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST")
            or (not text.strip())
        ):
            # Treated like a Claude refusal: recorded, counted, not raised.
            meta["refusal_category"] = str(getattr(block, "name", block)) if block else None
            meta["stop_reason"] = "refusal"
            return GeneratedAnswer(text="", meta=meta)
        answer = parse_answer(text, contexts)
        answer.meta = meta
        return answer


GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqGenerator:
    """Grounded answers from Groq (OpenAI-style chat completions, `groq` SDK).

    The API key is read by the SDK from the GROQ_API_KEY environment variable;
    nothing is stored in the repo. Same prompt contract as the other generators;
    temperature 0; transient errors (429, 5xx) are retried with backoff.
    """

    def __init__(
        self,
        model: str = GROQ_DEFAULT_MODEL,
        max_tokens: int = 8192,  # reasoning models spend tokens before the answer
        client=None,
        max_attempts: int = 6,
        base_delay: float = 2.0,
    ):
        if client is None:
            import groq  # lazy: only needed when Groq is actually used

            client = groq.Groq(max_retries=0)  # retries handled below, visibly
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.name = f"groq(model={model},temperature=0)"

    def _call_with_retry(self, **kwargs):
        import random
        import time

        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                code = getattr(e, "status_code", None)
                transient = code == 429 or (isinstance(code, int) and code >= 500)
                if not transient or attempt == self.max_attempts:
                    raise
                delay = min(self.base_delay * 2 ** (attempt - 1), 60) + random.uniform(0, 1)
                print(f"  Groq {code}; retry {attempt}/{self.max_attempts - 1} in {delay:.0f}s")
                time.sleep(delay)

    def generate(
        self, question: str, contexts: list[SearchResult], confidence: float | None = None
    ) -> GeneratedAnswer:
        response = self._call_with_retry(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question, contexts)},
            ],
        )
        choice = response.choices[0]
        usage = response.usage
        meta = {
            "model": response.model,
            "stop_reason": choice.finish_reason,
            "request_id": response.id,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        }
        text = choice.message.content or ""
        if choice.finish_reason == "content_filter":
            meta["stop_reason"] = "refusal"
            return GeneratedAnswer(text="", meta=meta)
        if not text.strip():
            # e.g. finish_reason "length": the token budget ran out before an answer.
            # Recorded as an empty, non-abstained answer; not counted as a refusal.
            meta["empty_answer"] = True
            return GeneratedAnswer(text="", meta=meta)
        answer = parse_answer(text, contexts)
        answer.meta = meta
        return answer
