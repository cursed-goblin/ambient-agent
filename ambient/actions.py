"""
OS action handlers for the Phase 1 rules layer (spec 4.15).

Every handler:
  * returns a short plain-English string to be SPOKEN back
  * respects config.DRY_RUN
  * never raises -- a missing binary returns a spoken explanation

No LLM is involved. These are called directly from a matched Intent.
"""

from __future__ import annotations

import datetime as _dt
import re
import shutil
import subprocess
import threading
import time
from typing import Optional

import config
from ambient.rules import Intent, clamp
from ambient.state import log_event

COMMAND_TIMEOUT = 15


# --------------------------------------------------------------------------
# subprocess helper
# --------------------------------------------------------------------------

def run(cmd: list[str], timeout: int = COMMAND_TIMEOUT) -> tuple[int, str]:
    """Run a command. Returns (returncode, combined output). Never raises."""
    if config.DRY_RUN:
        log_event("dry_run", command=" ".join(cmd))
        return 0, "[dry-run]"
    if not shutil.which(cmd[0]):
        return 127, f"{cmd[0]} is not installed"
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        log_event("command", command=" ".join(cmd), rc=proc.returncode)
        return proc.returncode, out.strip()
    except subprocess.TimeoutExpired:
        return 124, "command timed out"
    except Exception as exc:  # pragma: no cover
        return 1, str(exc)


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------

def _get_volume() -> Optional[int]:
    rc, out = run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
    if rc != 0:
        return None
    m = re.search(r"(\d{1,3})%", out)
    return int(m.group(1)) if m else None


def set_volume(intent: Intent) -> str:
    level = clamp(int(intent.slots["level"]))
    rc, out = run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"])
    return f"Volume {level} percent." if rc == 0 else f"Couldn't set volume: {out}"


def step_volume(intent: Intent) -> str:
    delta = int(intent.slots.get("delta", 10))
    current = _get_volume()
    if current is None:
        sign = "+" if delta >= 0 else "-"
        rc, out = run(["pactl", "set-sink-volume", "@DEFAULT_SINK@",
                       f"{sign}{abs(delta)}%"])
        return "Done." if rc == 0 else f"Couldn't change volume: {out}"
    target = clamp(current + delta)
    rc, out = run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{target}%"])
    return f"Volume {target} percent." if rc == 0 else f"Couldn't change volume: {out}"


def mute(intent: Intent) -> str:
    enabled = bool(intent.slots.get("enabled", True))
    rc, out = run(["pactl", "set-sink-mute", "@DEFAULT_SINK@",
                   "1" if enabled else "0"])
    if rc != 0:
        return f"Couldn't change mute: {out}"
    return "Muted." if enabled else "Unmuted."


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def _brightness_pct() -> Optional[int]:
    rc, out = run(["brightnessctl", "-m", "info"])
    if rc != 0:
        return None
    m = re.search(r"(\d{1,3})%", out)
    return int(m.group(1)) if m else None


def set_brightness(intent: Intent) -> str:
    level = clamp(int(intent.slots["level"]), 1, 100)
    rc, out = run(["brightnessctl", "set", f"{level}%"])
    return f"Brightness {level} percent." if rc == 0 else f"Couldn't set brightness: {out}"


def step_brightness(intent: Intent) -> str:
    delta = int(intent.slots.get("delta", 10))
    arg = f"{abs(delta)}%{'+' if delta >= 0 else '-'}"
    rc, out = run(["brightnessctl", "set", arg])
    if rc != 0:
        return f"Couldn't change brightness: {out}"
    now = _brightness_pct()
    return f"Brightness {now} percent." if now is not None else "Done."


def dark_mode(intent: Intent) -> str:
    enabled = bool(intent.slots.get("enabled", True))
    scheme = "prefer-dark" if enabled else "default"
    rc, out = run(["gsettings", "set", "org.gnome.desktop.interface",
                   "color-scheme", scheme])
    if rc != 0:
        return f"Couldn't change appearance: {out}"
    return "Dark mode on." if enabled else "Dark mode off."


# --------------------------------------------------------------------------
# Apps / windows
# --------------------------------------------------------------------------

APP_ALIASES = {
    "browser": "chromium", "chrome": "chromium", "chromium": "chromium",
    "firefox": "firefox", "terminal": "gnome-terminal", "files": "nautilus",
    "file manager": "nautilus", "settings": "gnome-control-center",
    "calculator": "gnome-calculator", "text editor": "gedit",
    "music": "rhythmbox", "code": "code", "editor": "code",
}


def _resolve_app(name: str) -> Optional[str]:
    name = (name or "").strip().lower()
    if not name:
        return None
    if name in APP_ALIASES:
        binary = APP_ALIASES[name]
        return binary if shutil.which(binary) else None
    candidate = name.replace(" ", "-")
    return candidate if shutil.which(candidate) else None


def open_app(intent: Intent) -> str:
    requested = intent.slots.get("app", "")
    binary = _resolve_app(requested)
    if not binary:
        return f"I don't know how to open {requested}."
    if config.DRY_RUN:
        log_event("dry_run", command=binary)
        return f"Would open {requested}."
    try:
        subprocess.Popen(
            [binary],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        log_event("app_open", app=requested, binary=binary)
        return f"Opening {requested}."
    except Exception as exc:
        return f"Couldn't open {requested}: {exc}"


def close_app(intent: Intent) -> str:
    requested = intent.slots.get("app", "")
    rc, _ = run(["wmctrl", "-c", requested])
    return f"Closing {requested}." if rc == 0 else f"Couldn't close {requested}."


def focus_app(intent: Intent) -> str:
    requested = intent.slots.get("app", "")
    rc, _ = run(["wmctrl", "-a", requested])
    return f"Switching to {requested}." if rc == 0 else f"I don't see {requested} open."


# --------------------------------------------------------------------------
# Timers (tier 0)
# --------------------------------------------------------------------------

class TimerService:
    def __init__(self, on_fire) -> None:
        self._on_fire = on_fire
        self._handle: Optional[threading.Timer] = None
        self._ends_at: Optional[float] = None
        self._label = ""

    def start(self, seconds: int, label: str = "") -> str:
        self.cancel(silent=True)
        self._ends_at = time.time() + seconds
        self._label = label
        self._handle = threading.Timer(seconds, self._fire)
        self._handle.daemon = True
        self._handle.start()
        log_event("timer_start", seconds=seconds)
        return f"Timer set for {_humanise(seconds)}."

    def _fire(self) -> None:
        self._ends_at = None
        log_event("timer_fire")
        try:
            self._on_fire("Your timer is done.")
        except Exception:
            pass

    def cancel(self, silent: bool = False) -> str:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
            self._ends_at = None
            if not silent:
                log_event("timer_cancel")
            return "Timer cancelled."
        return "There's no timer running."

    def remaining(self) -> str:
        if self._ends_at is None:
            return "There's no timer running."
        left = max(0, int(self._ends_at - time.time()))
        return f"{_humanise(left)} left."


def _humanise(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        if secs:
            return f"{minutes} minute{'s' if minutes != 1 else ''} {secs} seconds"
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minutes"


# --------------------------------------------------------------------------
# Read-only info
# --------------------------------------------------------------------------

def info_time(_intent: Intent) -> str:
    return "It's " + _dt.datetime.now().strftime("%-I:%M %p").lower() + "."


def info_date(_intent: Intent) -> str:
    return "Today is " + _dt.datetime.now().strftime("%A, %-d %B %Y") + "."


def info_battery(_intent: Intent) -> str:
    for path, label in (("/sys/class/power_supply/BAT0", "BAT0"),
                        ("/sys/class/power_supply/BAT1", "BAT1")):
        try:
            with open(f"{path}/capacity", encoding="utf-8") as fh:
                pct = fh.read().strip()
            with open(f"{path}/status", encoding="utf-8") as fh:
                status = fh.read().strip().lower()
            suffix = " and charging" if status == "charging" else ""
            return f"Battery is at {pct} percent{suffix}."
        except OSError:
            continue
    return "I couldn't read the battery -- this might be a desktop."


def info_disk(_intent: Intent) -> str:
    try:
        usage = shutil.disk_usage("/")
        free_gb = usage.free / 1024 ** 3
        total_gb = usage.total / 1024 ** 3
        pct = 100 * usage.used / usage.total
        return (f"{free_gb:.0f} gigabytes free of {total_gb:.0f}, "
                f"about {pct:.0f} percent used.")
    except Exception as exc:
        return f"Couldn't read disk usage: {exc}"


def info_memory(_intent: Intent) -> str:
    try:
        values = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                values[key] = int(rest.strip().split()[0])
        total = values["MemTotal"] / 1024 ** 2
        available = values["MemAvailable"] / 1024 ** 2
        return f"{available:.1f} gigabytes free of {total:.1f}."
    except Exception as exc:
        return f"Couldn't read memory: {exc}"


def info_wifi(_intent: Intent) -> str:
    rc, out = run(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"])
    if rc != 0:
        return "Couldn't check the network."
    for line in out.splitlines():
        if line.startswith("yes:"):
            return f"Connected to {line.split(':', 1)[1]}."
    return "Not connected to wifi."


# --------------------------------------------------------------------------
# Media (XF86 keys via playerctl)
# --------------------------------------------------------------------------

def _player(cmd: str, spoken: str) -> str:
    rc, _ = run(["playerctl", cmd])
    return spoken if rc == 0 else "No media player is running."


# --------------------------------------------------------------------------
# Tier 0 answers
# --------------------------------------------------------------------------

def tier0_math(intent: Intent) -> str:
    return f"That's {intent.slots['result']}."


def tier0_convert(intent: Intent) -> str:
    s = intent.slots
    return f"{s['value']} {s['from']} is {s['result']} {s['to']}."


# --------------------------------------------------------------------------
# Dispatch table
# --------------------------------------------------------------------------

def build_dispatch(timer_service: TimerService, panel=None) -> dict:
    """
    Map intent name -> handler(Intent) -> spoken string.

    `panel` is a placeholder for the Phase 2 overlay. When it is None the
    panel intents return a spoken acknowledgement so Phase 1 stays runnable.
    """

    def panel_stub(label: str):
        def handler(_intent: Intent) -> str:
            if panel is None:
                return f"Nothing on screen to {label} yet."
            return getattr(panel, label)()
        return handler

    return {
        "audio.set_volume": set_volume,
        "audio.step_volume": step_volume,
        "audio.mute": mute,
        "display.set_brightness": set_brightness,
        "display.step_brightness": step_brightness,
        "display.dark_mode": dark_mode,
        "app.open": open_app,
        "app.close": close_app,
        "app.focus": focus_app,
        "timer.start": lambda i: timer_service.start(int(i.slots.get("seconds", 60))),
        "timer.cancel": lambda _i: timer_service.cancel(),
        "timer.remaining": lambda _i: timer_service.remaining(),
        "info.time": info_time,
        "info.date": info_date,
        "info.battery": info_battery,
        "info.disk": info_disk,
        "info.memory": info_memory,
        "info.wifi": info_wifi,
        "media.play": lambda _i: _player("play", "Playing."),
        "media.pause": lambda _i: _player("pause", "Paused."),
        "media.next": lambda _i: _player("next", "Next track."),
        "media.previous": lambda _i: _player("previous", "Previous track."),
        "panel.next": panel_stub("next"),
        "panel.previous": panel_stub("previous"),
        "panel.close": panel_stub("close"),
        "panel.fullscreen": panel_stub("fullscreen"),
        "panel.select": panel_stub("select"),
        "tier0.math": tier0_math,
        "tier0.convert": tier0_convert,
    }
