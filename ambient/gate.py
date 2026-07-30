"""
Permission gate (spec 4.8) -- the ONE funnel every tool call passes through.

The agent loop is not allowed to touch the operating system directly. It asks
the gate, and the gate decides. That gives us four properties the model cannot
talk its way around, because they are enforced in Python:

1. **Risk tiers.** Every tool is declared SAFE, CAUTION or DANGER. SAFE runs
   silently. CAUTION runs and is audited. DANGER stops and asks a human.
2. **Forbidden patterns are refused, not confirmed.** A catastrophic argument
   (rm -rf /, mkfs, fork bomb) is rejected outright. There is no phrase the
   user or the model can say to proceed. Refusal is not a prompt instruction,
   it is a regex in front of the executor.
3. **No confirmation without a summary in the same turn.** An approval is only
   valid if the gate spoke what it was about to do, and only for 60 seconds.
   This is what stops "misheard command deleted something".
4. **Everything is written to audit.log** before and after execution --
   timestamp, tool, arguments, risk, decision, reason, result.

DRY_RUN is enforced here as well as in the handlers, deliberately. Two
independent checks on the thing that touches your machine is not redundancy
worth removing.

Note what is absent: there is no `run_command`, no `delete_path`, no
`install_package`. Those are not blocked by a prompt -- they do not exist in
the codebase. That is the strongest safety property available and it costs
nothing to keep.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Optional

import config
from ambient import tools as tools_mod
from ambient.state import log_event

# --- risk tiers (defined in risk.py so tools.py can import them too) ---
from ambient.risk import CAUTION, DANGER, SAFE  # noqa: E402,F401

# How long a spoken summary stays valid for confirmation.
APPROVAL_TTL_S = float(getattr(config, "APPROVAL_TTL_S", 60))

# Distinctive on purpose. Never bare "yes" or "okay" -- ambient conversation
# and TV audio trigger those constantly.
CONFIRM_PHRASES = tuple(
    getattr(config, "CONFIRM_PHRASES",
            ("confirm that", "go ahead", "do it now", "yes confirm"))
)
CANCEL_PHRASES = tuple(
    getattr(config, "CANCEL_PHRASES",
            ("cancel that", "never mind", "nevermind", "forget it", "stop that"))
)

# Catastrophic argument patterns. These are REFUSED, never offered for
# confirmation. Checked against every stringified argument value.
_FORBIDDEN = tuple(re.compile(p, re.I) for p in (
    r"rm\s+-[a-z]*[rf]",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r">\s*/dev/[sh]d",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bchmod\s+-R\s+777",
    r"curl[^|]*\|\s*(ba)?sh",
    r"wget[^|]*\|\s*(ba)?sh",
    r":\(\)\s*\{\s*:\|:",
    r"/etc/(passwd|shadow|sudoers)",
    r"\bsudo\b",
))


class Decision:
    """What the gate decided about one proposed tool call."""

    __slots__ = ("allowed", "risk", "needs_approval", "summary", "refusal")

    def __init__(self, allowed: bool, risk: str, needs_approval: bool = False,
                 summary: str = "", refusal: str = "") -> None:
        self.allowed = allowed
        self.risk = risk
        self.needs_approval = needs_approval
        self.summary = summary
        self.refusal = refusal

    def __repr__(self) -> str:
        return (f"Decision(allowed={self.allowed}, risk={self.risk}, "
                f"needs_approval={self.needs_approval})")


class _Pending:
    __slots__ = ("tool", "arguments", "reason", "risk", "summary", "created")

    def __init__(self, tool: str, arguments: dict, reason: str, risk: str,
                 summary: str) -> None:
        self.tool = tool
        self.arguments = arguments
        self.reason = reason
        self.risk = risk
        self.summary = summary
        self.created = time.monotonic()

    def expired(self) -> bool:
        return (time.monotonic() - self.created) > APPROVAL_TTL_S


def _readable(tool: str, arguments: dict) -> str:
    """Plain-English rendering of a call, for the spoken summary."""
    args = {k: v for k, v in (arguments or {}).items() if k != "reason"}
    if not args:
        return tool.replace("_", " ")
    parts = [f"{k.replace('_', ' ')} {v}" for k, v in args.items()]
    return tool.replace("_", " ") + " with " + ", ".join(parts)


class Gate:
    """
    Thread-safe. The UI thread, the voice loop and the timer service can all
    reach it, so pending state is locked.
    """

    def __init__(self, dry_run: Optional[bool] = None,
                 confirm_everything: Optional[bool] = None,
                 risk_map: Optional[dict] = None) -> None:
        self.dry_run = (getattr(config, "DRY_RUN", True)
                        if dry_run is None else dry_run)
        self.confirm_everything = (getattr(config, "CONFIRM_EVERYTHING", False)
                                   if confirm_everything is None
                                   else confirm_everything)
        self.risk_map = risk_map if risk_map is not None else tools_mod.RISK
        self._pending: Optional[_Pending] = None
        self._lock = threading.Lock()

    # -- audit ----------------------------------------------------------
    def _audit(self, **record: Any) -> None:
        record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            path = config.AUDIT_LOG
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            # An unwritable audit log must not stop the assistant, but it must
            # be loud in the event log.
            log_event("audit_write_failed", error=str(exc)[:200])

    # -- classification -------------------------------------------------
    def risk_of(self, tool: str) -> str:
        """Unknown tools are DANGER, not SAFE. Reject by default."""
        return self.risk_map.get(tool, DANGER)

    def forbidden_reason(self, arguments: dict) -> Optional[str]:
        for value in (arguments or {}).values():
            text = str(value)
            for pattern in _FORBIDDEN:
                if pattern.search(text):
                    return pattern.pattern
        return None

    def evaluate(self, tool: str, arguments: dict, reason: str = "") -> Decision:
        """Decide without executing. Pure function of the inputs + flags."""
        arguments = arguments or {}

        if tool not in self.risk_map:
            return Decision(False, DANGER,
                            refusal="I don't have a tool for that.")

        hit = self.forbidden_reason(arguments)
        if hit is not None:
            self._audit(tool=tool, arguments=arguments, risk=DANGER,
                        decision="refused_forbidden", pattern=hit,
                        reason=reason)
            log_event("gate_refused_forbidden", tool=tool, pattern=hit)
            return Decision(False, DANGER,
                            refusal="That's not something I will do.")

        risk = self.risk_of(tool)
        summary = _readable(tool, arguments)

        if risk == DANGER:
            return Decision(True, risk, needs_approval=True, summary=summary)
        if risk == CAUTION and self.confirm_everything:
            return Decision(True, risk, needs_approval=True, summary=summary)
        return Decision(True, risk)

    # -- execution ------------------------------------------------------
    def run(self, tool: str, arguments: dict, reason: str = "") -> str:
        """
        The single entry point. Returns text for the model (or the user).

        On DANGER, nothing executes: a pending approval is stored and the
        spoken summary is returned. main.py listens for a confirm phrase.
        """
        arguments = arguments or {}
        decision = self.evaluate(tool, arguments, reason)

        if not decision.allowed:
            return decision.refusal

        if decision.needs_approval:
            with self._lock:
                self._pending = _Pending(tool, arguments, reason,
                                         decision.risk, decision.summary)
            self._audit(tool=tool, arguments=arguments, risk=decision.risk,
                        decision="awaiting_approval", reason=reason)
            log_event("gate_awaiting_approval", tool=tool, risk=decision.risk)
            phrase = CONFIRM_PHRASES[0]
            return (f"That will {decision.summary}. "
                    f"Say '{phrase}' if you want me to.")

        return self._execute(tool, arguments, reason, decision.risk)

    def _execute(self, tool: str, arguments: dict, reason: str,
                 risk: str) -> str:
        if self.dry_run:
            self._audit(tool=tool, arguments=arguments, risk=risk,
                        decision="dry_run", reason=reason)
            log_event("gate_dry_run", tool=tool)
            return f"[dry run] would {_readable(tool, arguments)}"

        self._audit(tool=tool, arguments=arguments, risk=risk,
                    decision="executing", reason=reason)
        result = tools_mod.execute(tool, arguments)
        self._audit(tool=tool, arguments=arguments, risk=risk,
                    decision="executed", reason=reason, result=str(result)[:400])
        log_event("gate_executed", tool=tool, risk=risk)
        return result

    # -- approval -------------------------------------------------------
    def has_pending(self) -> bool:
        with self._lock:
            if self._pending is not None and self._pending.expired():
                self._audit(tool=self._pending.tool, risk=self._pending.risk,
                            decision="approval_expired")
                self._pending = None
            return self._pending is not None

    def pending_summary(self) -> Optional[str]:
        with self._lock:
            return self._pending.summary if self._pending else None

    def confirm(self) -> str:
        """Execute the pending call. Expired or absent approvals are refused."""
        with self._lock:
            pending = self._pending
            self._pending = None

        if pending is None:
            return "There's nothing waiting for confirmation."
        if pending.expired():
            self._audit(tool=pending.tool, risk=pending.risk,
                        decision="approval_expired")
            log_event("gate_approval_expired", tool=pending.tool)
            return "That confirmation timed out. Ask me again."

        self._audit(tool=pending.tool, arguments=pending.arguments,
                    risk=pending.risk, decision="approved",
                    reason=pending.reason)
        log_event("gate_approved", tool=pending.tool)
        return self._execute(pending.tool, pending.arguments,
                             pending.reason, pending.risk)

    def cancel(self) -> str:
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            return "Nothing to cancel."
        self._audit(tool=pending.tool, arguments=pending.arguments,
                    risk=pending.risk, decision="cancelled",
                    reason=pending.reason)
        log_event("gate_cancelled", tool=pending.tool)
        return "Cancelled."


def is_confirm_phrase(text: str) -> bool:
    lowered = (text or "").strip().lower().rstrip(".!")
    return any(p in lowered for p in CONFIRM_PHRASES)


def is_cancel_phrase(text: str) -> bool:
    lowered = (text or "").strip().lower().rstrip(".!")
    return any(p in lowered for p in CANCEL_PHRASES)
