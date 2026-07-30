"""
AI escalation (spec 4.6, 4.7, 4.16).

This is the ONLY place a model is allowed to speak. It runs after the
deterministic rules layer has already failed to match.

Stdlib only -- no `openai` package. Both Groq and Ollama expose the same
OpenAI-compatible /chat/completions endpoint, so one small urllib client
covers both and there is one less dependency to break.

The anti-hallucination design, in order of importance:

1. The model has NO tools and NO ability to act. It cannot open apps, change
   settings or buy anything. If a request needs an action, it must answer with
   REFUSE and the deterministic layer stays the only thing that touches the OS.
2. It is told to say it does not know, and the refusal is a valid answer.
3. Answers are capped at three sentences in code, not just in the prompt.
4. Temperature is low. Creative writing is not the job.
5. Any failure -- timeout, bad key, model down -- returns None, which the
   caller turns into the ordinary refusal line. It never invents a fallback.

NOTE (Phase 2): `Escalator` below is the old words-only path. The live path is
now `chat_with_tools` + ambient/agent.py, where the model calls tools itself.
Escalator is kept for --check-ai and as an offline fallback.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Optional

import config
from ambient.state import log_event, timer

REFUSE_TOKEN = "REFUSE"
USER_AGENT = "ambient-agent/0.2 (+https://github.com/cursed-goblin/ambient-agent)"

SYSTEM_PROMPT = """You are the fallback for a hands-free voice assistant on Linux.

A deterministic rules engine already handles all device control: volume,
brightness, opening and closing apps, timers, battery, disk, memory, wifi,
time, date, arithmetic and unit conversion. It could not match this request,
so it was passed to you.

You have NO tools. You cannot open apps, change settings, browse the web, buy
anything, or take any action whatsoever. You can only answer with words.

Rules, in priority order:
1. If the request asks you to DO something rather than answer something, reply
   with exactly: REFUSE
2. If answering would need current information you do not have -- prices, news,
   weather, stock, live status, anything after your training data -- reply with
   exactly: REFUSE
3. If you are not confident the answer is correct, reply with exactly: REFUSE
4. Otherwise answer in at most three short sentences, plain spoken English, no
   markdown, no lists, no emoji. It will be read aloud by a speech synthesiser.
5. Never claim to have done anything. Never say you are opening, setting,
   buying or checking something.

REFUSE is a correct and expected answer. Being limited is fine. Being
confidently wrong is not."""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN = re.compile(r"[*_`#>\[\]]")


class LlmError(RuntimeError):
    pass


class LlmClient:
    """Minimal OpenAI-compatible chat client over urllib."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 30.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list, temperature: float = 0.2,
             max_tokens: int = 160) -> str:
        if not self.base_url or not self.model:
            raise LlmError("no provider configured")
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise LlmError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LlmError(f"cannot reach {self.base_url}: {exc.reason}") from exc
        except (ValueError, TimeoutError) as exc:
            raise LlmError(str(exc)) from exc

        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"unexpected response shape: {str(payload)[:200]}") from exc

    def chat_with_tools(self, messages: list, tool_schemas: list,
                        temperature: float = 0.2,
                        max_tokens: int = 512) -> dict:
        """
        One round-trip of tool-calling chat.

        Returns {"content": str|None, "tool_calls": list}. The caller runs any
        tool calls and calls this again with the results appended.
        """
        if not self.base_url or not self.model:
            raise LlmError("no provider configured")
        payload_body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tool_schemas:
            payload_body["tools"] = tool_schemas
            payload_body["tool_choice"] = "auto"
        body = json.dumps(payload_body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise LlmError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LlmError(f"cannot reach {self.base_url}: {exc.reason}") from exc
        except (ValueError, TimeoutError) as exc:
            raise LlmError(str(exc)) from exc

        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"unexpected response: {str(payload)[:200]}") from exc
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls") or [],
        }

    def ping(self) -> tuple[bool, str]:
        """Cheap reachability check for --check and the setup wizard."""
        try:
            reply = self.chat(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                temperature=0.0,
                max_tokens=8,
            )
            return True, reply.strip()[:40]
        except LlmError as exc:
            return False, str(exc)


def _clean(text: str, max_sentences: int = 3) -> str:
    text = _MARKDOWN.sub("", (text or "").strip())
    text = " ".join(text.split())
    parts = [p for p in _SENTENCE_SPLIT.split(text) if p]
    return " ".join(parts[:max_sentences]).strip()


class Escalator:
    """
    Wraps the client with the refusal contract.

    `answer()` returns a spoken string, or None meaning "refuse". None is a
    perfectly normal outcome and the caller must handle it without complaint.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        self.provider = self.cfg.get("provider", "none")
        self.enabled = self.provider not in ("none", "", None)
        self.client = (
            LlmClient(
                base_url=self.cfg.get("base_url", ""),
                api_key=self.cfg.get("api_key", ""),
                model=self.cfg.get("model", ""),
            )
            if self.enabled
            else None
        )

    def answer(self, text: str) -> Optional[str]:
        if not self.enabled or self.client is None:
            return None

        clock = timer("llm")
        try:
            raw = self.client.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=config.LLM_TEMPERATURE,
            )
        except LlmError as exc:
            clock.stop(ok=False)
            log_event("llm_error", provider=self.provider, error=str(exc)[:300])
            return None
        clock.stop(ok=True, provider=self.provider)

        reply = _clean(raw, config.LLM_MAX_SENTENCES)

        if not reply or REFUSE_TOKEN in reply.upper():
            log_event("llm_refused", text=text[:120])
            return None

        # Belt and braces: the model must never imply it acted on the system.
        lowered = reply.lower()
        for claim in ("i've opened", "i have opened", "i've set", "i have set",
                      "i've changed", "opening now", "i just bought",
                      "i've ordered", "i have ordered"):
            if claim in lowered:
                log_event("llm_blocked_action_claim", reply=reply[:160])
                return None

        log_event("llm_answer", provider=self.provider, model=self.cfg.get("model"),
                  text=text[:120], reply=reply[:300])
        return reply


def load_escalator(cfg: dict) -> Escalator:
    esc = Escalator(cfg)
    log_event("escalator_loaded", provider=esc.provider, enabled=esc.enabled)
    return esc
