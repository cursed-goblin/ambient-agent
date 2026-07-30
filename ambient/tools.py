"""
Tool registry -- capabilities described as schemas, not as matched commands.

There is deliberately no intent matching anywhere in this file. The model reads
the user's words, picks a tool from SCHEMAS, and fills in the arguments. Adding
a capability means adding one schema and one handler; it never means writing
another regex.

Three things every tool declares:

- **a schema** in SCHEMAS, in OpenAI tool-calling format
- **a risk tier** in RISK, consumed by ambient/gate.py
- **a required `reason`**, a plain-English justification the model must supply.
  It is written to audit.log and shown to the user on any approval prompt. A
  model that cannot say why it is doing something does not get to do it.

Handlers never raise. A tool that explodes returns a sentence the model can
read and recover from, because an exception mid-loop would strand the user in
silence.

Execution safety lives in gate.py, not here. This module is the executor of
last resort; the DRY_RUN check below is a second independent guard, kept on
purpose.
"""

from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Optional

import config
from ambient.state import log_event

from ambient.risk import CAUTION, DANGER, SAFE  # noqa: F401  (re-export)

COMMAND_TIMEOUT = int(getattr(config, "COMMAND_TIMEOUT", 15))

SCHEMAS: list[dict] = []
RISK: dict[str, str] = {}
HANDLERS: dict[str, Callable[..., str]] = {}


def _tool(name: str, description: str, props: dict, required: list[str],
          risk: str) -> None:
    """Register one capability. `reason` is appended to every schema."""
    props = dict(props)
    props["reason"] = {
        "type": "string",
        "description": "Short plain-English reason you are doing this. "
                       "Shown to the user and written to the audit log.",
    }
    SCHEMAS.append({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(required) + ["reason"],
            },
        },
    })
    RISK[name] = risk


def _dry_run() -> bool:
    return bool(getattr(config, "DRY_RUN", True))


def _run(cmd: list[str] | str) -> tuple[bool, str]:
    """Run a shell command. Returns (ok, output). Never raises."""
    shell = isinstance(cmd, str)
    printable = cmd if shell else " ".join(cmd)
    if _dry_run():
        log_event("dry_run_command", command=printable[:200])
        return True, ""
    try:
        proc = subprocess.run(cmd, shell=shell, capture_output=True,
                              text=True, timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, "the command timed out"
    except (OSError, ValueError) as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "").strip()[:200]
    return True, (proc.stdout or "").strip()


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def _clamp(value: Any, low: int, high: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def _fmt_secs(total: int) -> str:
    total = max(0, int(total))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
    if minutes and seconds:
        return f"{minutes} minute{'s' if minutes != 1 else ''} {seconds} second{'s' if seconds != 1 else ''}"
    if minutes:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{seconds} second{'s' if seconds != 1 else ''}"


# ----------------------------------------------------------------------
# audio
# ----------------------------------------------------------------------
_SINK = "@DEFAULT_SINK@"


def h_set_volume(level: Any = 50, **_kw) -> str:
    level = _clamp(level, 0, 100)
    ok, err = _run(["pactl", "set-sink-volume", _SINK, f"{level}%"])
    return f"Volume set to {level} percent." if ok else f"Couldn't set volume: {err}"


def h_step_volume(direction: str = "up", amount: Any = 10, **_kw) -> str:
    amount = _clamp(amount, 1, 100)
    sign = "-" if str(direction).lower().startswith("d") else "+"
    ok, err = _run(["pactl", "set-sink-volume", _SINK, f"{sign}{amount}%"])
    word = "down" if sign == "-" else "up"
    return f"Volume {word} {amount} percent." if ok else f"Couldn't change volume: {err}"


def h_set_mute(muted: Any = True, **_kw) -> str:
    flag = "1" if muted in (True, "true", "True", 1, "1", "yes") else "0"
    ok, err = _run(["pactl", "set-sink-mute", _SINK, flag])
    if not ok:
        return f"Couldn't change mute: {err}"
    return "Muted." if flag == "1" else "Unmuted."


# ----------------------------------------------------------------------
# display
# ----------------------------------------------------------------------

def h_set_brightness(level: Any = 50, **_kw) -> str:
    level = _clamp(level, 1, 100)
    if not _dry_run() and not _have("brightnessctl"):
        return "brightnessctl isn't installed, so I can't change brightness."
    ok, err = _run(["brightnessctl", "set", f"{level}%"])
    return f"Brightness set to {level} percent." if ok else f"Couldn't set brightness: {err}"


def h_step_brightness(direction: str = "up", amount: Any = 10, **_kw) -> str:
    amount = _clamp(amount, 1, 100)
    down = str(direction).lower().startswith("d")
    arg = f"{amount}%-" if down else f"+{amount}%"
    ok, err = _run(["brightnessctl", "set", arg])
    word = "down" if down else "up"
    return f"Brightness {word} {amount} percent." if ok else f"Couldn't change brightness: {err}"


def h_set_dark_mode(enabled: Any = True, **_kw) -> str:
    on = enabled in (True, "true", "True", 1, "1", "yes")
    scheme = "prefer-dark" if on else "prefer-light"
    ok, err = _run(["gsettings", "set", "org.gnome.desktop.interface",
                    "color-scheme", scheme])
    if not ok:
        return f"Couldn't change the theme: {err}"
    return "Dark mode on." if on else "Dark mode off."


# ----------------------------------------------------------------------
# apps
# ----------------------------------------------------------------------
_BROWSER_APP = "chromium --app=https://"

_APP_ALIASES = {
    "browser": "chromium", "chrome": "chromium", "chromium": "chromium",
    "firefox": "firefox",
    "whatsapp": _BROWSER_APP + "web.whatsapp.com",
    "youtube": _BROWSER_APP + "youtube.com",
    "gmail": _BROWSER_APP + "mail.google.com",
    "email": _BROWSER_APP + "mail.google.com",
    "maps": _BROWSER_APP + "maps.google.com",
    "terminal": "gnome-terminal", "console": "gnome-terminal",
    "files": "nautilus", "file manager": "nautilus",
    "calculator": "gnome-calculator", "calc": "gnome-calculator",
    "settings": "gnome-control-center",
    "code": "code", "vscode": "code", "editor": "gedit", "notepad": "gedit",
    "music": "rhythmbox", "video": "totem",
}


def _resolve_app(name: str) -> str:
    key = (name or "").strip().lower()
    return _APP_ALIASES.get(key, key)


def h_open_app(name: str = "", **_kw) -> str:
    target = _resolve_app(name)
    if not target:
        return "I didn't catch which app to open."
    ok, err = _run(f"nohup {target} >/dev/null 2>&1 &")
    return f"Opening {name}." if ok else f"Couldn't open {name}: {err}"


def h_close_app(name: str = "", **_kw) -> str:
    target = _resolve_app(name).split()[0] if _resolve_app(name) else ""
    if not target:
        return "I didn't catch which app to close."
    ok, err = _run(["pkill", "-f", target])
    return f"Closed {name}." if ok else f"{name} didn't seem to be running."


def h_focus_app(name: str = "", **_kw) -> str:
    target = _resolve_app(name).split()[0] if _resolve_app(name) else ""
    if not target:
        return "I didn't catch which app to switch to."
    ok, err = _run(["wmctrl", "-a", target])
    return f"Switched to {name}." if ok else f"Couldn't find a {name} window."


def h_media_control(action: str = "play-pause", **_kw) -> str:
    allowed = {"play", "pause", "play-pause", "next", "previous", "stop"}
    act = str(action).lower().replace("_", "-")
    if act not in allowed:
        return f"I can only do: {', '.join(sorted(allowed))}."
    ok, err = _run(["playerctl", act])
    return f"Media {act}." if ok else "Nothing seems to be playing."


# ----------------------------------------------------------------------
# timers
# ----------------------------------------------------------------------
_TIMERS: dict[str, dict] = {}
_TIMER_LOCK = threading.Lock()
_TIMER_CALLBACK: Optional[Callable[[str], None]] = None


def init_timer_callback(callback: Callable[[str], None]) -> None:
    """main.py hands us the speak function so a timer can announce itself."""
    global _TIMER_CALLBACK
    _TIMER_CALLBACK = callback


def _fire(label: str) -> None:
    with _TIMER_LOCK:
        entry = _TIMERS.pop(label, None)
    if entry is None:
        return  # cancelled while we were waiting
    message = f"Your {label} is done." if label != "timer" else "Your timer is done."
    log_event("timer_fired", label=label)
    if _TIMER_CALLBACK is not None:
        try:
            _TIMER_CALLBACK(message)
        except Exception as exc:
            log_event("timer_callback_error", error=str(exc)[:200])


def h_start_timer(seconds: Any = 0, label: str = "timer", **_kw) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "I didn't catch how long to set it for."
    if total <= 0:
        return "I need a duration longer than zero."
    if total > 24 * 3600:
        return "I can only set timers up to 24 hours."
    label = (label or "timer").strip().lower()

    with _TIMER_LOCK:
        existing = _TIMERS.pop(label, None)
    if existing is not None:
        existing["handle"].cancel()

    handle = threading.Timer(total, _fire, args=(label,))
    handle.daemon = True
    with _TIMER_LOCK:
        _TIMERS[label] = {"handle": handle, "ends": time.monotonic() + total,
                          "total": total}
    handle.start()
    log_event("timer_started", label=label, seconds=total)
    return f"{_fmt_secs(total)} {label} started."


def h_get_timer_remaining(label: str = "", **_kw) -> str:
    with _TIMER_LOCK:
        if not _TIMERS:
            return "You don't have any timers running."
        key = (label or "").strip().lower()
        if key and key in _TIMERS:
            items = [(key, _TIMERS[key])]
        else:
            items = list(_TIMERS.items())
        parts = []
        for name, entry in items:
            left = int(entry["ends"] - time.monotonic())
            parts.append(f"{_fmt_secs(left)} left on your {name}")
    return "; ".join(parts) + "."


def h_cancel_timer(label: str = "", **_kw) -> str:
    with _TIMER_LOCK:
        if not _TIMERS:
            return "There's no timer to cancel."
        key = (label or "").strip().lower()
        if key and key in _TIMERS:
            targets = [key]
        else:
            targets = list(_TIMERS.keys())
        for name in targets:
            entry = _TIMERS.pop(name, None)
            if entry is not None:
                entry["handle"].cancel()
    log_event("timer_cancelled", labels=targets)
    if len(targets) == 1:
        return f"Cancelled your {targets[0]}."
    return f"Cancelled {len(targets)} timers."


# ----------------------------------------------------------------------
# read-only info
# ----------------------------------------------------------------------

def _battery() -> str:
    try:
        with open("/sys/class/power_supply/BAT0/capacity", encoding="utf-8") as fh:
            pct = fh.read().strip()
        return f"Battery is at {pct} percent."
    except OSError:
        return "I couldn't read the battery level."


def h_get_info(kind: str = "time", **_kw) -> str:
    kind = (kind or "time").strip().lower()
    now = _dt.datetime.now()

    if kind == "time":
        return "It's " + now.strftime("%-I:%M %p") + "."
    if kind == "date":
        return "It's " + now.strftime("%A, %-d %B %Y") + "."
    if kind == "battery":
        return _battery()
    if kind == "volume":
        ok, out = _run(["pactl", "get-sink-volume", _SINK])
        if ok and "%" in out:
            return "Volume is " + out.split("/")[1].strip() + "."
        return "I couldn't read the volume."
    if kind == "disk":
        used = shutil.disk_usage("/")
        free_gb = used.free / (1024 ** 3)
        return f"You have {free_gb:.0f} gigabytes free."
    if kind == "memory":
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                lines = {l.split(":")[0]: l for l in fh}
            avail_kb = int(lines["MemAvailable"].split()[1])
            return f"About {avail_kb / (1024 ** 2):.1f} gigabytes of memory free."
        except (OSError, KeyError, IndexError, ValueError):
            return "I couldn't read the memory usage."
    return f"I don't track '{kind}'."


# ----------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------
_LEVEL = {"type": "integer", "description": "Target percentage, 0 to 100."}
_DIRECTION = {"type": "string", "enum": ["up", "down"]}
_AMOUNT = {"type": "integer", "description": "Percentage points to change by."}

_tool("set_volume", "Set the speaker volume to an absolute percentage.",
      {"level": _LEVEL}, ["level"], CAUTION)
_tool("step_volume", "Turn the volume up or down by a relative amount.",
      {"direction": _DIRECTION, "amount": _AMOUNT}, ["direction"], CAUTION)
_tool("set_mute", "Mute or unmute the speakers.",
      {"muted": {"type": "boolean"}}, ["muted"], CAUTION)
_tool("set_brightness", "Set screen brightness to an absolute percentage.",
      {"level": _LEVEL}, ["level"], CAUTION)
_tool("step_brightness", "Make the screen brighter or dimmer by a relative amount.",
      {"direction": _DIRECTION, "amount": _AMOUNT}, ["direction"], CAUTION)
_tool("set_dark_mode", "Turn the desktop dark theme on or off.",
      {"enabled": {"type": "boolean"}}, ["enabled"], CAUTION)
_tool("open_app",
      "Launch an application or a web app. Accepts everyday names like "
      "'whatsapp', 'youtube', 'browser', 'calculator', 'settings'.",
      {"name": {"type": "string", "description": "Everyday name of the app."}},
      ["name"], CAUTION)
_tool("close_app",
      "Force-quit an application. This can lose unsaved work, so it always "
      "asks the user before running.",
      {"name": {"type": "string"}}, ["name"], DANGER)
_tool("focus_app", "Bring an already-running application's window to the front.",
      {"name": {"type": "string"}}, ["name"], CAUTION)
_tool("start_timer",
      "Start a countdown timer. Convert the spoken duration to seconds "
      "yourself, e.g. 'ten minutes' becomes 600.",
      {"seconds": {"type": "integer", "description": "Total duration in seconds."},
       "label": {"type": "string",
                 "description": "Optional name, e.g. 'pasta'. Defaults to 'timer'."}},
      ["seconds"], CAUTION)
_tool("get_timer_remaining",
      "Check how much time is left on a running timer.",
      {"label": {"type": "string", "description": "Optional timer name."}},
      [], SAFE)
_tool("cancel_timer", "Cancel a running timer.",
      {"label": {"type": "string"}}, [], CAUTION)
_tool("media_control", "Control media playback in any running player.",
      {"action": {"type": "string",
                  "enum": ["play", "pause", "play-pause", "next", "previous", "stop"]}},
      ["action"], CAUTION)
_tool("get_info",
      "Read a piece of system or clock information. Use this instead of "
      "guessing the time, date or battery level.",
      {"kind": {"type": "string",
                "enum": ["time", "date", "battery", "volume", "disk", "memory"]}},
      ["kind"], SAFE)

HANDLERS.update({
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
})


def execute(name: str, arguments: dict) -> str:
    """
    Run one tool. Never raises -- the agent loop must always get a sentence
    back, even on failure, or the user is left in silence.

    Call this through ambient.gate.Gate.run(), not directly.
    """
    handler = HANDLERS.get(name)
    if handler is None:
        return f"I don't have a tool called {name}."
    kwargs = {k: v for k, v in (arguments or {}).items() if k != "reason"}
    try:
        return handler(**kwargs)
    except TypeError as exc:
        log_event("tool_bad_args", tool=name, error=str(exc)[:200])
        return f"Those arguments didn't work for {name}."
    except Exception as exc:
        log_event("tool_error", tool=name, error=str(exc)[:200])
        return f"{name} failed: {str(exc)[:120]}"
