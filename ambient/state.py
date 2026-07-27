"""
Mode state machine + audit logging (spec 4.2, 4.8).

Three modes, one at a time:
  ambient : wake word only, camera off      <- default
  assist  : mic open, camera on, gestures   <- short bursts only
  manual  : mic off, camera off             <- plain computer

Keyboard and mouse are ALWAYS live. Nothing here ever grabs an input device.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import config


class Mode(str, Enum):
    AMBIENT = "ambient"
    ASSIST = "assist"
    MANUAL = "manual"


class Activity(str, Enum):
    """Drives the overlay animation in Phase 2."""
    IDLE = "idle"
    WAKE = "wake"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ACTING = "acting"


@dataclass
class AgentState:
    mode: Mode = Mode.AMBIENT
    activity: Activity = Activity.IDLE
    mic_hot: bool = False
    camera_hot: bool = False
    last_reply: str = ""
    partial_transcript: str = ""
    status_line: str = ""
    _listeners: list = field(default_factory=list, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ---- observers (Phase 2 overlay subscribes here) --------------------
    def subscribe(self, fn: Callable[["AgentState"], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def _notify(self) -> None:
        for fn in list(self._listeners):
            try:
                fn(self)
            except Exception:  # a broken UI must never kill the agent
                pass

    # ---- mutators -------------------------------------------------------
    def set_activity(self, activity: Activity, status: str = "") -> None:
        with self._lock:
            self.activity = activity
            self.status_line = status
        self._notify()

    def set_partial(self, text: str) -> None:
        with self._lock:
            self.partial_transcript = text
        self._notify()

    def set_mode(self, mode: Mode) -> None:
        with self._lock:
            if mode == self.mode:
                return
            self.mode = mode
            self.mic_hot = mode is not Mode.MANUAL
            self.camera_hot = mode is Mode.ASSIST
            self.activity = Activity.IDLE
        log_event("mode_change", mode=mode.value,
                  mic_hot=self.mic_hot, camera_hot=self.camera_hot)
        self._notify()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode.value,
                "activity": self.activity.value,
                "mic_hot": self.mic_hot,
                "camera_hot": self.camera_hot,
                "status_line": self.status_line,
                "partial": self.partial_transcript,
                "last_reply": self.last_reply,
            }


# --------------------------------------------------------------------------
# Audit log (spec 4.8) -- every request, decision and result, timestamped.
# --------------------------------------------------------------------------

_audit_lock = threading.Lock()


def log_event(event: str, **fields) -> None:
    config.ensure_dirs()
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event}
    record.update(fields)
    line = json.dumps(record, default=str, ensure_ascii=False)
    with _audit_lock:
        try:
            with open(config.AUDIT_LOG, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


# --------------------------------------------------------------------------
# Latency instrumentation -- barge-in is a hard requirement, so measure it.
# --------------------------------------------------------------------------

@dataclass
class Timer:
    label: str
    start: float = field(default_factory=time.perf_counter)

    def stop(self, **extra) -> float:
        ms = (time.perf_counter() - self.start) * 1000
        log_event("latency", label=self.label, ms=round(ms, 1), **extra)
        return ms


def timer(label: str) -> Timer:
    return Timer(label)


STATE = AgentState(mode=Mode(config.START_MODE))


def current() -> AgentState:
    return STATE


def parse_mode(name: str) -> Optional[Mode]:
    try:
        return Mode(name)
    except ValueError:
        return None
