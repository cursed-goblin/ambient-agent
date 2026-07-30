"""
The agent loop -- the model decides, the gate permits, the tools act.

This replaced the old regex intent table. There is no per-command code here.
The model receives the user's words plus every schema in tools.SCHEMAS, picks
what to call, and fills in the arguments itself. "open whatsapp" was never
programmed; the model reads it and chooses open_app.

What is still hard-coded, deliberately:

- **Complexity caps** (spec 4.7), enforced in Python rather than asked for in
  the prompt: at most 6 tool calls per task, at most 2 retries of the same
  tool, and a wall-clock deadline. A model cannot talk its way past a while
  loop counter.
- **Every call goes through the gate.** This loop has no access to tools.execute
  and never calls a shell. It can only ask ambient.gate.Gate.
- **Approval short-circuits the loop.** If the gate wants a human, we stop
  immediately and return the spoken summary. We do not continue reasoning as
  though the action succeeded, which is the classic way agents end up lying.
- **Dry-run results are returned verbatim.** The model does not get to
  paraphrase them. See _DRY_PREFIX below for why.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import config
from ambient import gate as gate_mod
from ambient import tools as tools_mod
from ambient.llm import LlmError
from ambient.state import log_event

MAX_TOOL_CALLS = int(getattr(config, "MAX_TOOL_CALLS", 6))
MAX_RETRIES_PER_TOOL = 2
TASK_TIMEOUT_S = float(getattr(config, "TASK_TIMEOUT_S", 90))

# Marker the gate puts in front of anything it declined to actually perform.
_DRY_PREFIX = "[dry run]"

SYSTEM_PROMPT = """You are a hands-free voice assistant running on the user's \
Linux computer. Your reply is read aloud by a speech synthesiser.

You have tools. Use them. When the user asks for something a tool can do, call \
the tool -- do not describe how to do it and do not ask permission first, the \
system handles permission itself.

How to behave:
- Work out the arguments yourself. "ten minutes" is 600 seconds. "turn it down \
a bit" is step_volume down by about 10. "open whatsapp" is open_app with the \
name whatsapp.
- Every tool call needs a short `reason`. Write it in plain English; the user \
may see it.
- Use get_info rather than guessing the time, date or battery level. You do \
not know them.
- After a tool runs, tell the user what happened in one short sentence, based \
only on what the tool actually returned.
- If a tool returns an error, say so plainly. Never claim something worked \
when it did not.

Never substitute one tool for another. Your tools do exactly what their \
descriptions say and nothing more. If the user asks for something none of your \
tools can do, say plainly that you cannot do it yet -- do not call the nearest \
plausible tool and describe its result as if it achieved what was asked.

In particular you have NO tools for files or folders, no shell or terminal \
access, no way to install software, and no way to read or send messages. \
Opening an application is not the same as doing something inside it. If asked \
to create a folder, delete a file, run a command or send a message, say that \
is not something you can do.

Style: spoken English, short, no markdown, no lists, no emoji, no stage \
directions. One or two sentences unless asked for detail."""


def _model_error(exc: Exception) -> str:
    """Turn a transport failure into something worth hearing out loud."""
    text = str(exc).lower()
    if "401" in text or "invalid api key" in text:
        return "My API key was rejected. Check it in Settings."
    if "1010" in text or "403" in text:
        return ("The AI provider refused the connection. "
                "This is usually the network blocking it, not the key.")
    if "429" in text or "rate limit" in text:
        return "I'm being rate limited. Try again in a moment."
    if "cannot reach" in text or "timed out" in text or "timeout" in text:
        return "I can't reach the AI model right now."
    return "The AI model failed to respond."


class AgentLoop:
    def __init__(self, client=None, gate=None, history_turns: int = 6) -> None:
        self.client = client
        self.gate = gate if gate is not None else gate_mod.Gate()
        self.history_turns = history_turns
        self._history: list[dict] = []

    # -- history --------------------------------------------------------
    def _remember(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        keep = self.history_turns * 2
        if len(self._history) > keep:
            del self._history[:-keep]

    def reset(self) -> None:
        self._history.clear()

    # -- main -----------------------------------------------------------
    def handle(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        if self.client is None:
            return ("No AI model is connected yet. "
                    "Open Settings and add a Groq API key.")

        deadline = time.monotonic() + TASK_TIMEOUT_S
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": text})

        failures: dict[str, int] = {}
        calls_made = 0
        dry_results: list[str] = []

        while calls_made < MAX_TOOL_CALLS:
            if time.monotonic() > deadline:
                log_event("agent_timeout", text=text[:120])
                return ("That took longer than I allow myself. "
                        "I've stopped where I got to.")

            try:
                reply = self.client.chat_with_tools(messages, tools_mod.SCHEMAS)
            except LlmError as exc:
                log_event("agent_llm_error", error=str(exc)[:300])
                return _model_error(exc)
            except Exception as exc:
                log_event("agent_llm_crash", error=str(exc)[:300])
                return _model_error(exc)

            tool_calls = reply.get("tool_calls") or []
            content = (reply.get("content") or "").strip()

            # No tools wanted: this is the spoken answer.
            if not tool_calls:
                if not content:
                    return "I'm not sure what to do with that."
                self._remember("user", text)
                self._remember("assistant", content)
                return content

            messages.append({
                "role": "assistant",
                "content": reply.get("content"),
                "tool_calls": tool_calls,
            })

            for call in tool_calls:
                calls_made += 1
                function = (call.get("function") or {})
                name = function.get("name") or ""
                raw_args = function.get("arguments") or "{}"

                if isinstance(raw_args, str):
                    try:
                        arguments = json.loads(raw_args)
                    except ValueError:
                        arguments = {}
                else:
                    arguments = dict(raw_args)

                reason = str(arguments.get("reason") or "")
                log_event("agent_tool_call", tool=name, reason=reason[:120])

                # Everything funnels through the gate. Always.
                result = self.gate.run(name, arguments, reason)

                # The gate wants a human. Stop the loop; do not pretend.
                if self.gate.has_pending():
                    return result

                result_text = str(result)

                # Nothing actually happened. Keep the model away from it:
                # given a dry-run result and a request it could not fulfil, a
                # model will write a confident past-tense sentence over the
                # top of both. Observed in the wild -- "create a folder abhi"
                # came back as "Settings was opened to create a new folder"
                # when no folder existed and Settings had not even launched.
                if result_text.startswith(_DRY_PREFIX):
                    dry_results.append(result_text)

                lowered = result_text.lower()
                if "failed" in lowered or "couldn't" in lowered:
                    failures[name] = failures.get(name, 0) + 1

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or name,
                    "name": name,
                    "content": result_text,
                })

                if failures.get(name, 0) >= MAX_RETRIES_PER_TOOL:
                    log_event("agent_tool_giving_up", tool=name)
                    return (f"I tried {name.replace('_', ' ')} twice and it "
                            "kept failing, so I've stopped.")

            # Report dry runs ourselves, in the gate's own words. The model is
            # not asked to summarise an action that did not occur.
            if dry_results:
                log_event("agent_dry_run_reported", count=len(dry_results))
                joined = " ".join(dry_results)
                return (f"Safety mode is on, so nothing was changed. {joined}. "
                        "Turn safety mode off in Settings to let me act.")

        log_event("agent_cap_reached", text=text[:120])
        return ("This is more complex than I can handle in one go. "
                "Try asking for one thing at a time.")
