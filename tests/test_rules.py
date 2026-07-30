"""
Unit tests for the rules layer.

These run anywhere -- no mic, no GPU, no models, no network. Run with:
    python3 -m pytest tests/ -q
or without pytest:
    python3 tests/test_rules.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ambient import rules  # noqa: E402


def check(utterance, expected_intent, **expected_slots):
    intent = rules.match(utterance)
    assert intent is not None, f"no match for {utterance!r}"
    assert intent.name == expected_intent, (
        f"{utterance!r} -> {intent.name}, expected {expected_intent}"
    )
    for key, value in expected_slots.items():
        assert intent.slots.get(key) == value, (
            f"{utterance!r} slot {key}={intent.slots.get(key)!r}, expected {value!r}"
        )
    return intent


# ---------------------------------------------------------------- control

def test_stop_is_control():
    for phrase in ("stop", "cancel", "shut up", "never mind", "stop talking"):
        intent = check(phrase, "control.stop")
        assert rules.is_control(intent), phrase


def test_control_bypasses_gate():
    assert rules.match("stop").risk == rules.CONTROL


# ---------------------------------------------------------------- volume

def test_volume_absolute_digits():
    check("set the volume to 40", "audio.set_volume", level=40)
    check("volume 100", "audio.set_volume", level=100)


def test_volume_absolute_words():
    check("set volume to fifty", "audio.set_volume", level=50)
    check("volume to seventy five percent", "audio.set_volume", level=75)


def test_volume_clamped():
    check("volume 300", "audio.set_volume", level=100)


def test_volume_steps():
    assert rules.match("volume up").slots["delta"] > 0
    assert rules.match("turn it down").slots["delta"] < 0
    assert rules.match("louder").name == "audio.step_volume"


def test_mute():
    check("mute", "audio.mute", enabled=True)
    check("unmute", "audio.mute", enabled=False)


# ------------------------------------------------------------ brightness

def test_brightness():
    check("set brightness to 40", "display.set_brightness", level=40)
    check("brightness to half", "display.set_brightness", level=50)
    assert rules.match("dim the screen").slots["delta"] < 0


def test_dark_mode():
    check("turn on dark mode", "display.dark_mode", enabled=True)
    check("light mode", "display.dark_mode", enabled=False)


# ------------------------------------------------------------------ apps

def test_open_app():
    check("open browser", "app.open", app="browser")
    check("launch terminal", "app.open", app="terminal")
    check("switch to firefox", "app.focus", app="firefox")


def test_open_with_filler_words():
    check("hey can you please open the calculator", "app.open")


# ---------------------------------------------------------------- timers

def test_timer_minutes():
    check("set a timer for 10 minutes", "timer.start", seconds=600)
    check("timer for five minutes", "timer.start", seconds=300)


def test_timer_seconds_and_hours():
    check("set a timer for 30 seconds", "timer.start", seconds=30)
    check("set a timer for 2 hours", "timer.start", seconds=7200)


def test_timer_cancel_and_query():
    check("cancel the timer", "timer.cancel")
    check("how long left on the timer", "timer.remaining")


# ------------------------------------------------------------------ info

def test_info_intents():
    check("what's the time", "info.time")
    check("what's today's date", "info.date")
    check("battery level", "info.battery")
    check("disk usage", "info.disk")
    check("memory free", "info.memory")
    check("wifi status", "info.wifi")


def test_info_bare_words():
    # Regression: a real session refused a bare "battery" because the
    # qualifier word ("level", "percent", ...) was mandatory in the pattern.
    check("battery", "info.battery")
    check("disk", "info.disk")
    check("memory", "info.memory")
    check("ram", "info.memory")
    check("storage left", "info.disk")


# ------------------------------------------------------------- small talk

def test_smalltalk():
    # "hey" is a filler word to normalise(), so this also guards the path
    # where the whole utterance normalises to an empty string.
    check("hey", "smalltalk.greeting")
    check("hello", "smalltalk.greeting")
    check("good morning", "smalltalk.greeting")
    check("are you there", "smalltalk.presence")
    check("how are you", "smalltalk.how_are_you")
    check("thanks", "smalltalk.thanks")
    check("what can you do", "smalltalk.help")
    check("who are you", "smalltalk.identity")
    check("bye", "smalltalk.bye")


def test_smalltalk_does_not_hijack_commands():
    check("hey can you please open the calculator", "app.open")
    check("hi set volume to 40", "audio.set_volume", level=40)
    assert rules.match("help me book a flight to tokyo") is None


# ----------------------------------------------------------------- modes

def test_mode_switching():
    check("go to sleep", "mode.manual")
    check("gesture mode", "mode.assist")
    check("wake up", "mode.ambient")


# ----------------------------------------------------------------- panel

def test_panel_navigation():
    check("next", "panel.next")
    check("go back", "panel.previous")
    check("close", "panel.close")
    check("full screen", "panel.fullscreen")
    check("number three", "panel.select", index=3)


# ------------------------------------------------------ tier 0 (spec 4.16)

def test_tier0_math():
    intent = check("what's 12 * 8", "tier0.math")
    assert intent.slots["result"] == 96
    assert rules.match("45 + 55").slots["result"] == 100


def test_tier0_math_phrasings():
    # Whisper transcribes these all slightly differently.
    for phrase in ("what is 12 * 8", "whats 12 * 8", "calculate 12 * 8",
                   "how much is 12 * 8", "compute 12 * 8"):
        intent = rules.match(phrase)
        assert intent is not None, phrase
        assert intent.name == "tier0.math", phrase
        assert intent.slots["result"] == 96, phrase


def test_tier0_math_rejects_nonsense():
    assert rules.match("what's the meaning of life") is None


def test_tier0_conversion():
    intent = check("convert 10 km to miles", "tier0.convert")
    assert abs(intent.slots["result"] - 6.214) < 0.01
    intent = check("100 celsius in fahrenheit", "tier0.convert")
    assert abs(intent.slots["result"] - 212) < 0.01


# ------------------------------------------------- reject-by-default rule

def test_unknown_returns_none():
    for phrase in (
        "book me a flight to tokyo next tuesday",
        "write a poem about the sea",
        "what do you think about quantum physics",
        "asdkjhasd kjahsd",
        "",
        "   ",
    ):
        assert rules.match(phrase) is None, phrase


def test_incomplete_level_command_does_not_guess():
    # "set the volume" with no number must NOT silently pick a value.
    intent = rules.match("set the volume")
    assert intent is None or "level" in intent.slots


# ------------------------------------------------------------ normalisation

def test_normalise():
    assert rules.normalise("  Hey, PLEASE  turn ON  dark mode!! ") == "turn on dark mode"


def test_words_to_number():
    assert rules.words_to_number("forty five") == 45
    assert rules.words_to_number("seventy") == 70
    assert rules.words_to_number("half") == 50
    assert rules.words_to_number("no numbers here") is None


# ---------------------------------------------------------------- runner

def _run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append((name, str(exc)))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, repr(exc)))
            print(f"  ERROR {name}: {exc!r}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
