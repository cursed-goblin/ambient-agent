"""
Tool registry.

This is the ONLY place where we describe what the assistant can do. We do not
write intent-matching code. We describe each capability as a JSON schema, hand
the list to the model, and the model decides which one to call and with what
arguments.

Adding a new capability = adding one schema + one handler here. Nothing else.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from typing import Any, Callable, Optional

import config

COMMAND_TIMEOUT = getattr(config, "COMMAND_TIMEOUT", 15)


def _dry_run() -> bool:
    return bool(getattr(config, "DRY_RUN", True))


def _run(cmd: str) -> str:
    """Run a shell command. Respects DRY_RUN."""
    if _dry_run():
        return f"[dry-run] would run: {cmd}"
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {COMMAND_TIMEOUT}s"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return f"error (exit {proc.returncode}): {err or out or 'no output'}"
    return out or "done"


# --------------------------------------------------------------------------
# timers -- held in-process so the model can query them
# --------------------------------------------------------------------------
_TIMERS: dict[str, float] = {}
_TIMER_CB: Optional[Callable[[str], None]] = None


def init_timer_callback(cb: Callable[[str], None]) -> None:
    global _TIMER_CB
    _TIMER_CB = cb


def _fmt_secs(secs: float) -> str:
    secs = int(max(0, secs))
    if secs < 60:
        return f"{secs} second{'s' if secs != 1 else ''}"
    mins, rem = divmod(secs, 60)
    if mins < 60:
        return f"{mins} minute{'s' if mins != 1 else ''}" + (
            f" {rem} seconds" if rem else "")
    hrs, mins = divmod(mins, 60)
    return f"{hrs} hour{'s' if hrs != 1 else ''}" + (
        f" {mins} minutes" if mins else "")


# --------------------------------------------------------------------------
# app launching
# --------------------------------------------------------------------------
_APP_ALIASES = {
    "browser": "chromium",
    "chrome": "chromium",
    "whatsapp": "chromium --app=https://web.whatsapp.com",
    "youtube": "chromium --app=https://youtube.com",
    "gmail": "chromium --app=https://mail.google.com",
    "maps": "chromium --app=https://maps.google.com",
    "terminal": "gnome-terminal",
    "files": "nautilus",
    "calculator": "gnome-calculator",
    "settings": "gnome-control-center",
    "code": "code",
    "vscode": "code",
    "notepad": "gedit",
    "editor": "gedit",
}


def _resolve_app(name: str) -> str:
    key = (name or "").strip().lower()
    return _APP_ALIASES.get(key, shlex.quote(key))


# --------------------------------------------------------------------------
# handlers -- each returns a short string describing the RESULT
# --------------------------------------------------------------------------
def h_set_volume(level: int) -> str:
    level = max(0, min(100, int(level)))
    _run(f"pactl set-sink-volume @DEFAULT_SINK@ {level}%")
    return f"volume set to {level} percent"


def h_step_volume(direction: str, amount: int = 10) -> str:
    amount = max(1, min(50, int(amount)))
    sign = "+" if str(direction).lower() in ("up", "increase", "raise") else "-"
    _run(f"pactl set-sink-volume @DEFAULT_SINK@ {sign}{amount}%")
    return f"volume {'up' if sign == '+' else 'down'} {amount} percent"


def h_set_mute(muted: bool) -> str:
    _run(f"pactl set-sink-mute @DEFAULT_SINK@ {'1' if muted else '0'}")
    return "muted" if muted else "unmuted"


def h_set_brightness(level: int) -> str:
    level = max(1, min(100, int(level)))
    _run(f"brightnessctl set {level}%")
    return f"brightness set to {level} percent"


def h_step_brightness(direction: str, amount: int = 10) -> str:
    amount = max(1, min(50, int(amount)))
    up = str(direction).lower() in ("up", "increase", "raise", "brighter")
    _run(f"brightnessctl set {amount}%{'+' if up else '-'}")
    return f"brightness {'up' if up else 'down'} {amount} percent"


def h_set_dark_mode(enabled: bool) -> str:
    scheme = "prefer-dark" if enabled else "prefer-light"
    _run(f"gsettings set org.gnome.desktop.interface color-scheme {scheme}")
    return f"{'dark' if enabled else 'light'} mode on"


def h_open_app(name: str) -> str:
    cmd = _resolve_app(name)
    _run(f"nohup {cmd} >/dev/null 2>&1 &")
    return f"opening {name}"


def h_close_app(name: str) -> str:
    base = _resolve_app(name).split()[0]
    _run(f"pkill -f {shlex.quote(base)}")
    return f"closed {name}"


def h_focus_app(name: str) -> str:
    _run(f"wmctrl -a {shlex.quote(name)}")
    return f"switched to {name}"


def h_start_timer(seconds: int, label: str = "timer") -> str:
    seconds = max(1, int(seconds))
    _TIMERS[label] = time.monotonic() + seconds
    return f"{label} set for {_fmt_secs(seconds)}"


def h_get_timer_remaining(label: str = "timer") -> str:
    if not _TIMERS:
        return "no timers running"
    if label not in _TIMERS:
        label = next(iter(_TIMERS))
    left = _TIMERS[label] - time.monotonic()
    if left <= 0:
        _TIMERS.pop(label, None)
        return f"{label} already finished"
    return f"{_fmt_secs(left)} left on {label}"


def h_cancel_timer(label: str = "timer") -> str:
    if not _TIMERS:
        return "no timers to cancel"
    if label not in _TIMERS:
        label = next(iter(_TIMERS))
    _TIMERS.pop(label, None)
    return f"{label} cancelled"


def h_media_control(action: str) -> str:
    action = str(action).lower()
    mapping = {
        "play": "play", "pause": "pause", "playpause": "play-pause",
        "toggle": "play-pause", "next": "next", "previous": "previous",
        "prev": "previous", "stop": "stop",
    }
    verb = mapping.get(action, "play-pause")
    _run(f"playerctl {verb}")
    return f"media {verb}"


def h_get_info(kind: str) -> str:
    kind = str(kind).lower()
    if kind == "time":
        return time.strftime("%I:%M %p").lstrip("0")
    if kind == "date":
        return time.strftime("%A, %B %d")
    if kind == "battery":
        out = _run("cat /sys/class/power_supply/BAT0/capacity")
        return f"battery at {out} percent" if out.isdigit() else out
    if kind == "volume":
        return _run("pactl get-sink-volume @DEFAULT_SINK@")
    if kind == "disk":
        return _run("df -h / | tail -1 | awk '{print $4\" free of \"$2}'")
    if kind == "memory":
        return _run("free -h | awk 'NR==2{print $7\" available\"}'")
    return f"unknown info kind: {kind}"


HANDLERS: dict[str, Callable[..., str]] = {
    "set_volume": h_set_volume,
    "step_volume": h_step_volume,
    "set_mute": h_set_mute,
    "set_brightness": h_set_brightness,
    "step_brightness": h_step_brightness,
    "set_dark_mode": h_set_dark_mode,
    "open_app": h_open_app,
    "close_app": h_close_app,
    "focus_app": h_focus_app,
    "start_timer": h_start_timer,
    "get_timer_remaining": h_get_timer_remaining,
    "cancel_timer": h_cancel_timer,
    "media_control": h_media_control,
    "get_info": h_get_info,
}


def _tool(name: str, desc: str, props: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


SCHEMAS: list[dict] = [
    _tool("set_volume", "Set system volume to an absolute percentage.",
          {"level": {"type": "integer", "description": "0-100"}}, ["level"]),
    _tool("step_volume", "Turn the volume up or down by a relative amount.",
          {"direction": {"type": "string", "enum": ["up", "down"]},
           "amount": {"type": "integer", "description": "percent, default 10"}},
          ["direction"]),
    _tool("set_mute", "Mute or unmute system audio.",
          {"muted": {"type": "boolean"}}, ["muted"]),
    _tool("set_brightness", "Set screen brightness to an absolute percentage.",
          {"level": {"type": "integer", "description": "1-100"}}, ["level"]),
    _tool("step_brightness", "Make the screen brighter or dimmer.",
          {"direction": {"type": "string", "enum": ["up", "down"]},
           "amount": {"type": "integer"}}, ["direction"]),
    _tool("set_dark_mode", "Switch the desktop between dark and light theme.",
          {"enabled": {"type": "boolean"}}, ["enabled"]),
    _tool("open_app",
          "Open/launch an application or website. Accepts common names like "
          "whatsapp, youtube, browser, terminal, calculator, code, gmail.",
          {"name": {"type": "string"}}, ["name"]),
    _tool("close_app", "Close/quit a running application.",
          {"name": {"type": "string"}}, ["name"]),
    _tool("focus_app", "Switch to / focus an already-open application window.",
          {"name": {"type": "string"}}, ["name"]),
    _tool("start_timer",
          "Start a countdown timer. Convert any spoken duration to seconds.",
          {"seconds": {"type": "integer"},
           "label": {"type": "string", "description": "e.g. 'pasta', 'timer'"}},
          ["seconds"]),
    _tool("get_timer_remaining",
          "Check how much time is left on a running timer.",
          {"label": {"type": "string"}}, []),
    _tool("cancel_timer", "Cancel a running timer.",
          {"label": {"type": "string"}}, []),
    _tool("media_control", "Control media playback.",
          {"action": {"type": "string",
                      "enum": ["play", "pause", "toggle", "next",
                               "previous", "stop"]}}, ["action"]),
    _tool("get_info",
          "Read a piece of system or clock information.",
          {"kind": {"type": "string",
                    "enum": ["time", "date", "battery", "volume",
                             "disk", "memory"]}}, ["kind"]),
]


def execute(name: str, arguments: dict[str, Any]) -> str:
    """Run one tool by name. Never raises -- returns an error string instead."""
    fn = HANDLERS.get(name)
    if fn is None:
        return f"error: no such tool '{name}'"
    try:
        return fn(**(arguments or {}))
    except TypeError as exc:
        return f"error: bad arguments for {name}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"error running {name}: {exc}"
