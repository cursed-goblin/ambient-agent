"""
Central configuration. Every tunable lives here -- nothing hardcoded in modules.

Override any value with an env var, either AMBIENT_-prefixed or bare, e.g.
    AMBIENT_WHISPER_MODEL=small.en ./run.sh
    WHISPER_MODEL=base.en CONFIRM_EVERYTHING=1 ./run.sh
"""

from __future__ import annotations

import os
from pathlib import Path


def _env(key: str, default):
    # Accept both AMBIENT_WHISPER_MODEL and WHISPER_MODEL. The prefixed form is
    # what run.sh and the README use; the bare form stays supported for quick
    # one-off overrides on the command line.
    raw = os.environ.get("AMBIENT_" + key)
    if raw is None:
        raw = os.environ.get(key)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError:
            return default
    return raw


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "var")))
MODEL_DIR = Path(_env("MODEL_DIR", str(ROOT / "models")))
AUDIT_LOG = DATA_DIR / "audit.log"

# --------------------------------------------------------------------------
# Audio  (spec 4.3)
# --------------------------------------------------------------------------
SAMPLE_RATE = 16_000          # everything downstream expects 16k mono
CHANNELS = 1
DTYPE = "int16"
FRAME_MS = 32                 # Silero VAD wants 512 samples @ 16k = 32ms
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# Leave as None to use the PipeWire default (which should be the AEC source
# created by setup_audio.sh). Set to a device name substring to pin it.
INPUT_DEVICE = _env("INPUT_DEVICE", None)
OUTPUT_DEVICE = _env("OUTPUT_DEVICE", None)

# --------------------------------------------------------------------------
# Wake word
# --------------------------------------------------------------------------
WAKE_ENABLED = _env("WAKE_ENABLED", True)
WAKE_MODEL = _env("WAKE_MODEL", "hey_jarvis")   # openWakeWord bundled model
WAKE_THRESHOLD = _env("WAKE_THRESHOLD", 0.55)
# After a wake word or a reply, keep the mic open this long for a follow-up
# so the user does not have to say the wake word for every turn.
FOLLOW_UP_WINDOW_S = _env("FOLLOW_UP_WINDOW_S", 8.0)

# --------------------------------------------------------------------------
# VAD / barge-in  (spec 4.3 -- target < 300ms)
# --------------------------------------------------------------------------
VAD_THRESHOLD = _env("VAD_THRESHOLD", 0.5)
# Frames of speech required to declare "the user is talking".
# 3 frames x 32ms = ~96ms of speech before we cut our own audio.
BARGE_IN_FRAMES = _env("BARGE_IN_FRAMES", 3)
# Frames of speech required to *start* an utterance when we are silent.
SPEECH_START_FRAMES = _env("SPEECH_START_FRAMES", 2)
# Trailing silence that ends an utterance. 25 x 32ms = 800ms.
SPEECH_END_FRAMES = _env("SPEECH_END_FRAMES", 25)
MAX_UTTERANCE_S = _env("MAX_UTTERANCE_S", 15.0)
# While our own TTS is playing, require a slightly higher bar so residual
# echo that survives AEC cannot self-interrupt.
BARGE_IN_VAD_THRESHOLD = _env("BARGE_IN_VAD_THRESHOLD", 0.7)

# --------------------------------------------------------------------------
# STT  (spec 4.4)
# --------------------------------------------------------------------------
# tiny.en / base.en run fine on CPU -- start here, no GPU needed.
WHISPER_MODEL = _env("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = _env("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = _env("WHISPER_COMPUTE", "int8")
WHISPER_BEAM = _env("WHISPER_BEAM", 1)
PARTIAL_INTERVAL_S = _env("PARTIAL_INTERVAL_S", 0.6)
EMIT_PARTIALS = _env("EMIT_PARTIALS", True)

# Domain vocabulary biasing (spec 4.4). Materially improves recognition of
# these specific terms. Keep it short -- long prompts slow decoding.
WHISPER_PROMPT = _env(
    "WHISPER_PROMPT",
    "Flipkart, RedBus, UPI, brightness, dark mode, wifi, timer, "
    "volume, Chromium, screenshot, gesture mode, ambient mode.",
)

# --------------------------------------------------------------------------
# TTS  (spec 4.5)
# --------------------------------------------------------------------------
PIPER_BIN = _env("PIPER_BIN", "piper")
PIPER_VOICE = _env("PIPER_VOICE", str(MODEL_DIR / "piper" / "en_US-amy-medium.onnx"))
PIPER_SAMPLE_RATE = _env("PIPER_SAMPLE_RATE", 22_050)
# Playback chunk. Small enough to cut cleanly on barge-in (spec 4.3).
TTS_CHUNK_MS = _env("TTS_CHUNK_MS", 200)
TTS_ENABLED = _env("TTS_ENABLED", True)

# --------------------------------------------------------------------------
# Safety  (spec 4.8) -- Phase 1 stubs, fully consumed in Phase 4
# --------------------------------------------------------------------------
DRY_RUN = _env("DRY_RUN", False)
CONFIRM_EVERYTHING = _env("CONFIRM_EVERYTHING", False)
# Distinctive confirmation phrases only. Never bare "yes"/"okay" -- ambient
# conversation and TV audio trigger those constantly.
CONFIRM_PHRASES = ("confirm that", "go ahead", "do it now", "yes confirm")
DENY_PHRASES = ("cancel that", "no stop", "dont do it", "do not do it")

# --------------------------------------------------------------------------
# Modes  (spec 4.2)
# --------------------------------------------------------------------------
START_MODE = _env("START_MODE", "ambient")   # ambient | assist | manual
CAMERA_IDLE_TIMEOUT_S = _env("CAMERA_IDLE_TIMEOUT_S", 15.0)

# --------------------------------------------------------------------------
# Escalation  (Phase 4 -- off in Phase 1)
# --------------------------------------------------------------------------
LLM_ENABLED = _env("LLM_ENABLED", False)
# Swap this one line to move between cloud dev and local production.
LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = _env("LLM_API_KEY", "ollama")
LLM_MODEL = _env("LLM_MODEL", "qwen2.5:7b-instruct")
LLM_TEMPERATURE = _env("LLM_TEMPERATURE", 0.2)
MAX_TOOL_CALLS = _env("MAX_TOOL_CALLS", 6)
MAX_TOOL_RETRIES = _env("MAX_TOOL_RETRIES", 2)
TASK_TIMEOUT_S = _env("TASK_TIMEOUT_S", 90)
ROUTER_CONFIDENCE_FLOOR = _env("ROUTER_CONFIDENCE_FLOOR", 0.6)

REFUSAL_LINE = "Sorry, I can't do that yet."


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
