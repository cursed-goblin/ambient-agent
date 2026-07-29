"""
Deterministic rules layer -- the 80% path.

ZERO dependencies. Pure stdlib. This is intentional:
  * it must never fail
  * it must be unit-testable without a mic, a GPU, or a model
  * it must answer in under ~5ms

The LLM is NOT involved here. If nothing matches, we return None and the
caller decides whether to escalate (Phase 4) or refuse.

See spec section 4.6 (Intent router) and 4.16 tier 0.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

# --------------------------------------------------------------------------
# Risk levels (spec 4.8). The gate in Phase 4 consumes these.
# --------------------------------------------------------------------------
SAFE = "SAFE"
CAUTION = "CAUTION"
DANGER = "DANGER"
CONTROL = "CONTROL"  # special: never gated, never routed to a model (stop/cancel)


@dataclass
class Intent:
    name: str
    slots: dict = field(default_factory=dict)
    risk: str = SAFE
    # Confidence is 1.0 for rules -- an exact pattern matched. The fuzzy
    # embedding router (Phase 4) is the thing that produces < 1.0.
    confidence: float = 1.0
    utterance: str = ""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Intent({self.name}, {self.slots}, {self.risk})"


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

_FILLER = re.compile(
    r"\b(please|kindly|could you|can you|would you|hey|okay|ok|um|uh|just)\b"
)
_PUNCT = re.compile(r"[^\w\s%\-\.]")
_WS = re.compile(r"\s+")

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fourty": 40,  # common Whisper misspelling
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100,
    "half": 50, "quarter": 25, "full": 100, "max": 100, "maximum": 100,
    "min": 0, "minimum": 0, "mid": 50, "middle": 50,
}


def normalise(text: str) -> str:
    """Lowercase, strip accents/punctuation/filler, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = _FILLER.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


# Math needs its operators, which normalise() strips. Keep them here.
_PUNCT_MATH = re.compile(r"[^\w\s%\-\.\+\*/\(\)]")


def normalise_math(text: str) -> str:
    """Like normalise(), but preserves + - * / ( ) so try_math can see them."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT_MATH.sub(" ", text.lower().strip())
    return _WS.sub(" ", text).strip()


def words_to_number(text: str) -> Optional[int]:
    """'forty five' -> 45, 'seventy' -> 70, 'half' -> 50. Digits win."""
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        return int(m.group(1))

    tokens = [t for t in text.split() if t in _NUM_WORDS]
    if not tokens:
        return None

    total = 0
    for tok in tokens:
        val = _NUM_WORDS[tok]
        if val == 100 and total:
            total *= 100
        else:
            total += val
    return total if total <= 1000 else None


def clamp(value: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------
# Rule table
# --------------------------------------------------------------------------
# Each rule: (regex, intent_name, risk, slot_builder)
# Order matters -- first match wins, so put specific before general.
# --------------------------------------------------------------------------

RuleSlots = Callable[[re.Match, str], dict]


def _no_slots(_m: re.Match, _t: str) -> dict:
    return {}


def _level(_m: re.Match, text: str) -> dict:
    n = words_to_number(text)
    return {"level": clamp(n)} if n is not None else {}


def _step(sign: int, default: int = 10) -> RuleSlots:
    def build(_m: re.Match, text: str) -> dict:
        n = words_to_number(text)
        return {"delta": sign * (n if n is not None else default)}
    return build


def _app(m: re.Match, _t: str) -> dict:
    return {"app": (m.groupdict().get("app") or "").strip()}


def _bool(value: bool) -> RuleSlots:
    def build(_m: re.Match, _t: str) -> dict:
        return {"enabled": value}
    return build


def _duration(m: re.Match, text: str) -> dict:
    n = words_to_number(text)
    unit = (m.groupdict().get("unit") or "minute").rstrip("s")
    factor = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600}
    return {"seconds": (n or 1) * factor.get(unit, 60)}


def _index(m: re.Match, text: str) -> dict:
    n = words_to_number(m.groupdict().get("idx") or text)
    return {"index": n} if n is not None else {}


RULES: list[tuple[str, str, str, RuleSlots]] = [
    # ---- CONTROL: highest priority, never gated, never sent to a model ----
    (r"^(stop|cancel|abort|shut up|quiet|never mind|nevermind|halt)$",
     "control.stop", CONTROL, _no_slots),
    (r"\b(stop|cancel) (it|that|this|talking|listening)\b",
     "control.stop", CONTROL, _no_slots),

    # ---- Mode switching (spec 4.2) ----
    (r"\b(go to sleep|sleep now|manual mode|stop listening|mic off)\b",
     "mode.manual", CONTROL, _no_slots),
    (r"\b(assist mode|gesture mode|turn on (the )?camera|watch my hand)\b",
     "mode.assist", CONTROL, _no_slots),
    (r"\b(ambient mode|listen again|wake up)\b",
     "mode.ambient", CONTROL, _no_slots),

    # ---- Volume ----
    (r"\b(mute|silence)\b", "audio.mute", CAUTION, _bool(True)),
    (r"\b(unmute|sound on)\b", "audio.mute", CAUTION, _bool(False)),
    (r"\bvolume (to |at |)(\d{1,3}|\w+) ?(percent|%)?$",
     "audio.set_volume", CAUTION, _level),
    (r"\b(set |)volume (to|at) \b", "audio.set_volume", CAUTION, _level),
    (r"\b(volume up|louder|turn it up|increase (the )?volume)\b",
     "audio.step_volume", CAUTION, _step(+1)),
    (r"\b(volume down|quieter|softer|turn it down|lower (the )?volume|decrease (the )?volume)\b",
     "audio.step_volume", CAUTION, _step(-1)),

    # ---- Brightness ----
    (r"\bbrightness (to |at |)(\d{1,3}|\w+) ?(percent|%)?$",
     "display.set_brightness", CAUTION, _level),
    (r"\b(set |)brightness (to|at) \b",
     "display.set_brightness", CAUTION, _level),
    (r"\b(brighter|brightness up|increase brightness)\b",
     "display.step_brightness", CAUTION, _step(+10)),
    (r"\b(dimmer|dim the screen|brightness down|decrease brightness)\b",
     "display.step_brightness", CAUTION, _step(-10)),

    # ---- Appearance ----
    (r"\b(dark mode on|enable dark mode|turn on dark mode|go dark)\b",
     "display.dark_mode", CAUTION, _bool(True)),
    (r"\b(dark mode off|disable dark mode|turn off dark mode|light mode)\b",
     "display.dark_mode", CAUTION, _bool(False)),

    # ---- Apps / windows ----
    (r"\b(open|launch|start|run) (?P<app>[a-z0-9 \-\.]+?)$",
     "app.open", CAUTION, _app),
    (r"\b(close|quit|exit) (?P<app>[a-z0-9 \-\.]+?)$",
     "app.close", CAUTION, _app),
    (r"\b(switch to|focus|bring up) (?P<app>[a-z0-9 \-\.]+?)$",
     "app.focus", CAUTION, _app),

    # ---- Timers (tier 0, spec 4.16) ----
    (r"\b(cancel|stop) (the |my |)timer\b", "timer.cancel", SAFE, _no_slots),
    (r"\b(how long|time left|remaining|how much time)\b[a-z ]{0,25}\btimer\b",
     "timer.remaining", SAFE, _no_slots),
    (r"\b(set |start |)(a |)timer (for |of |)\b.*?(?P<unit>seconds?|secs?|minutes?|mins?|hours?)\b",
     "timer.start", SAFE, _duration),
    (r"\b(remind me in|wake me in)\b.*?(?P<unit>seconds?|minutes?|mins?|hours?)\b",
     "timer.start", SAFE, _duration),

    # ---- Read-only system info (SAFE, spec 4.15) ----
    (r"\b(what.?s the |)time( is it|)\b", "info.time", SAFE, _no_slots),
    (r"\b(what.?s |)(today.?s |the |)date\b", "info.date", SAFE, _no_slots),
    (r"\b(battery|charge)( (status|level|percent|percentage|left))?\b"
     r"|\bpower (status|level|percent|percentage|left)\b",
     "info.battery", SAFE, _no_slots),
    (r"\b(disk|storage)( (usage|space|free|left))?\b", "info.disk", SAFE, _no_slots),
    (r"\b(memory|ram)( (usage|free|left|available))?\b", "info.memory", SAFE, _no_slots),
    (r"\b(wifi|wi fi|network|internet) (status|connection|connected)\b",
     "info.wifi", SAFE, _no_slots),

    # ---- Media ----
    (r"\b(play|resume)( (the |)(music|song|video|audio))?$", "media.play", CAUTION, _no_slots),
    (r"\b(pause|hold)( (the |)(music|song|video|audio))?$", "media.pause", CAUTION, _no_slots),
    (r"\b(next|skip) (track|song)\b", "media.next", CAUTION, _no_slots),
    (r"\b(previous|last) (track|song)\b", "media.previous", CAUTION, _no_slots),

    # ---- Panel navigation (spec 4.9 -- same vocabulary everywhere) ----
    (r"^(next|forward)$", "panel.next", SAFE, _no_slots),
    (r"^(previous|back|go back)$", "panel.previous", SAFE, _no_slots),
    (r"^(close|dismiss|done|hide)( (it|that|the panel))?$", "panel.close", SAFE, _no_slots),
    (r"\b(full ?screen|show me (it|that) bigger|zoom in)\b", "panel.fullscreen", SAFE, _no_slots),
    (r"\b(number|item|option) (?P<idx>\d{1,2}|\w+)\b", "panel.select", SAFE, _index),
    (r"\b(repeat|say (that|it) again|what did you say)\b", "speech.repeat", SAFE, _no_slots),
]

_COMPILED = [(re.compile(p), name, risk, slots) for p, name, risk, slots in RULES]


# --------------------------------------------------------------------------
# Tier 0 local deterministic answers (spec 4.16) -- no internet, no model
# --------------------------------------------------------------------------

_MATH = re.compile(r"^[\d\s\.\+\-\*/x%\(\)]+$")
_MATH_PHRASE = re.compile(
    r"\b(what\s*(is|are|s)|whats|calculate|compute|how much is|equals?)"
    r"\b(?P<expr>[\d\s\.\+\-\*/x%\(\)]+)$"
)


def try_math(text: str) -> Optional[Intent]:
    """Safe arithmetic only. No eval of arbitrary input."""
    candidate = text
    m = _MATH_PHRASE.search(text)
    if m:
        candidate = m.group("expr")

    candidate = candidate.replace("x", "*").strip()
    if not candidate or not _MATH.match(candidate):
        return None
    if not re.search(r"[\+\-\*/]", candidate):
        return None

    try:
        # Restricted: only digits and operators survived the regex above.
        result = eval(candidate, {"__builtins__": {}}, {})  # noqa: S307
    except Exception:
        return None
    if isinstance(result, float):
        result = round(result, 4)
    return Intent("tier0.math", {"expression": candidate, "result": result},
                  SAFE, 1.0, text)


_CONVERSIONS = {
    ("km", "miles"): 0.621371, ("miles", "km"): 1.60934,
    ("kg", "pounds"): 2.20462, ("pounds", "kg"): 0.453592,
    ("kg", "lbs"): 2.20462, ("lbs", "kg"): 0.453592,
    ("cm", "inches"): 0.393701, ("inches", "cm"): 2.54,
    ("celsius", "fahrenheit"): None, ("fahrenheit", "celsius"): None,
}

_CONV_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<src>km|miles|kg|pounds|lbs|cm|inches|celsius|fahrenheit)"
    r".{0,12}?(?P<dst>km|miles|kg|pounds|lbs|cm|inches|celsius|fahrenheit)"
)


def try_conversion(text: str) -> Optional[Intent]:
    m = _CONV_RE.search(text)
    if not m:
        return None
    value = float(m.group("value"))
    src, dst = m.group("src"), m.group("dst")
    if src == dst:
        return None

    if (src, dst) == ("celsius", "fahrenheit"):
        result = value * 9 / 5 + 32
    elif (src, dst) == ("fahrenheit", "celsius"):
        result = (value - 32) * 5 / 9
    else:
        factor = _CONVERSIONS.get((src, dst))
        if factor is None:
            return None
        result = value * factor

    return Intent(
        "tier0.convert",
        {"value": value, "from": src, "to": dst, "result": round(result, 3)},
        SAFE, 1.0, text,
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def match(raw_text: str) -> Optional[Intent]:
    """
    Return an Intent if a rule or tier-0 answer matches, else None.

    None means "escalate or refuse" -- it is NEVER an error. Per spec 4.6,
    reject-by-default is the desired behaviour.
    """
    text = normalise(raw_text)
    if not text:
        return None

    for pattern, name, risk, build_slots in _COMPILED:
        m = pattern.search(text)
        if not m:
            continue
        intent = Intent(name, build_slots(m, text), risk, 1.0, text)
        # A level/step command with no parseable number is not a match --
        # better to fall through than to guess a value.
        if name.endswith(("set_volume", "set_brightness")) and "level" not in intent.slots:
            continue
        if name == "app.open" and not intent.slots.get("app"):
            continue
        return intent

    # Tier 0: deterministic local answers (spec 4.16). Math is checked
    # against a normalisation that keeps operators intact.
    intent = try_math(normalise_math(raw_text))
    if intent:
        return intent
    intent = try_conversion(text)
    if intent:
        return intent

    return None


def is_control(intent: Optional[Intent]) -> bool:
    """Control intents bypass the permission gate and the model entirely."""
    return intent is not None and intent.risk == CONTROL
