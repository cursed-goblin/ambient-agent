"""
The agent loop.

This replaces regex intent matching. The flow is:

    user text -> model (with tool schemas) -> model picks tool(s)
              -> we execute -> results back to model -> model speaks

We never decide what the user meant. The model does. Our job is only to
execute safely and keep the loop bounded.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from ambient import llm, tools

MAX_TOOL_CALLS = 6

SYSTEM_PROMPT = """You are Ambient, a hands-free assistant running on the \
user's own Linux computer.

You control the machine by calling tools. When the user asks for something you \
have a tool for, CALL THE TOOL -- do not describe how to do it, do not ask for \
confirmation, just do it. You may call several tools in a row if a request \
needs it.

Interpret natural, messy speech generously. Transcription is imperfect, so \
infer intent from context:
  "whatsapp buddy"        -> open_app("whatsapp")
  "timer check"           -> get_timer_remaining()
  "make it louder"        -> step_volume("up")
  "too bright"            -> step_brightness("down")
  "quarter hour timer"    -> start_timer(900)

Convert spoken durations to seconds yourself ("ten minutes" -> 600).

If no tool fits, just answer conversationally -- you are also a normal \
assistant and can chat, do arithmetic, and answer questions.

You are speaking out loud. Replies must be ONE short sentence, plain spoken \
English, no markdown, no lists, no emoji. Confirm what you did, briefly.
"""

# Substrings we refuse to let the model push into a shell, no matter what.
_DANGER = [
    "rm -rf", "dd if=", "mkfs", "> /dev", "shutdown", "reboot",
    "chmod -R 777", "curl | sh", "wget | sh", "fork bomb", ":(){ :|:",
    "/etc/passwd", "/etc/shadow",
]


def _is_safe(arguments: dict) -> bool:
    blob = json.dumps(arguments or {}).lower()
    return not any(bad in blob for bad in _DANGER)


class AgentLoop:
    """Owns one conversation with the model."""

    def __init__(self, client: Optional[llm.LlmClient],
                 speak_callback: Optional[Callable[[str], None]] = None,
                 history_turns: int = 6) -> None:
        self.client = client
        self.speak_callback = speak_callback
        self.history_turns = history_turns
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def handle(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if self.client is None:
            return self._offline_fallback(text)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history[-self.history_turns * 2:])
        messages.append({"role": "user", "content": text})

        used_a_tool = False

        for _ in range(MAX_TOOL_CALLS):
            try:
                reply = self.client.chat_with_tools(messages, tools.SCHEMAS)
            except llm.LlmError as exc:
                return self._model_error(exc)

            calls = reply.get("tool_calls") or []
            content = (reply.get("content") or "").strip()

            if not calls:
                final = content or ("Done." if used_a_tool else
                                    "I didn't catch that.")
                self._remember(text, final)
                return final

            # Record the assistant's tool-call turn verbatim.
            messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": calls,
            })

            for call in calls:
                used_a_tool = True
                fn = call.get("function", {})
                name = fn.get("name", "")
                raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {}

                if not _is_safe(args):
                    result = "error: refused for safety"
                else:
                    result = tools.execute(name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", name),
                    "name": name,
                    "content": str(result),
                })

        final = "That took too many steps."
        self._remember(text, final)
        return final

    # ------------------------------------------------------------------
    def _remember(self, user_text: str, reply: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        self.history = self.history[-self.history_turns * 2:]

    def _model_error(self, exc: Exception) -> str:
        detail = str(exc)
        if "401" in detail or "invalid_api_key" in detail:
            return "My API key was rejected. Check it in Settings."
        if "403" in detail or "1010" in detail:
            return "The AI provider blocked the request. Try another network."
        if "429" in detail:
            return "Rate limited. Give it a moment."
        if "cannot reach" in detail:
            return "I can't reach the AI provider. Check your connection."
        return "The AI request failed."

    def _offline_fallback(self, text: str) -> str:
        return ("No AI model is connected, so I can't understand that yet. "
                "Open Settings and add a Groq API key.")

    def reset(self) -> None:
        self.history.clear()
