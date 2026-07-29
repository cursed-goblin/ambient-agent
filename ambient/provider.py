"""
AI provider selection (spec 4.7 escalation).

Asked once, remembered in var/llm.json (chmod 600, gitignored so an API key
never reaches a commit).

Three choices, all handled through one OpenAI-compatible interface so the rest
of the code does not know or care which one is active:

  1. Groq    cloud, free tier, very fast. Needs an API key.
             NOT private: escalated text leaves your machine.
  2. Ollama  local, fully private, needs a machine that can hold the model.
  3. None    rules only -- the Phase 1 behaviour.

Audio never leaves the machine in any of these. Whisper is always local. Only
the transcribed text of an *escalated* request is sent, and only in Groq mode.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import config
from ambient.state import log_event

CONFIG_PATH = config.DATA_DIR / "llm.json"

PRESETS = {
    "groq": {
        "provider": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "api_key": "",
        "private": False,
    },
    "ollama": {
        "provider": "ollama",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b-instruct",
        "api_key": "ollama",
        "private": True,
    },
    "none": {
        "provider": "none",
        "base_url": "",
        "model": "",
        "api_key": "",
        "private": True,
    },
}


def load() -> Optional[dict]:
    """Load saved provider config. Env vars win, so CI can override it."""
    env_provider = os.environ.get("AMBIENT_LLM_PROVIDER")
    if env_provider:
        cfg = dict(PRESETS.get(env_provider, PRESETS["none"]))
        cfg["base_url"] = os.environ.get("AMBIENT_LLM_BASE_URL", cfg["base_url"])
        cfg["model"] = os.environ.get("AMBIENT_LLM_MODEL", cfg["model"])
        cfg["api_key"] = os.environ.get("AMBIENT_LLM_API_KEY", cfg["api_key"])
        return cfg

    if not CONFIG_PATH.exists():
        return None
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log_event("provider_load_failed", error=str(exc)[:200])
        return None


def save(cfg: dict) -> None:
    config.ensure_dirs()
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    try:
        os.chmod(CONFIG_PATH, 0o600)      # the file may hold an API key
    except OSError:
        pass
    log_event("provider_saved", provider=cfg.get("provider"),
              model=cfg.get("model"))


def describe(cfg: Optional[dict]) -> str:
    if not cfg or cfg.get("provider") in (None, "none"):
        return "no AI model (rules only)"
    privacy = "local, private" if cfg.get("private") else "cloud, NOT private"
    return f"{cfg['provider']} / {cfg['model']} ({privacy})"


def wizard() -> dict:
    """Interactive one-time chooser. Returns the saved config."""
    print()
    print("=" * 66)
    print("  Which AI model should handle things the rules cannot?")
    print("=" * 66)
    print()
    print("  1) Groq API      cloud, free tier, fast, no GPU needed.")
    print("                   Needs an API key from console.groq.com/keys")
    print("                   Escalated text leaves your machine.")
    print()
    print("  2) Local Ollama  fully private, nothing leaves the machine.")
    print("                   Needs Ollama running and a pulled model.")
    print("                   A 7B model wants ~8GB VRAM or ~10GB RAM.")
    print()
    print("  3) Neither       rules only. Anything unrecognised is refused.")
    print()
    print("  Either way: the wake word, mic, and speech recognition stay")
    print("  local. Audio is never uploaded.")
    print()

    choice = ""
    while choice not in ("1", "2", "3"):
        try:
            choice = input("  Choose 1, 2 or 3: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            choice = "3"

    if choice == "3":
        cfg = dict(PRESETS["none"])
        save(cfg)
        print("\n  Rules only. Re-run with --setup to change this.\n")
        return cfg

    if choice == "1":
        cfg = dict(PRESETS["groq"])
        key = ""
        while not key:
            try:
                key = input("  Paste your Groq API key (gsk_...): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled -- falling back to rules only.\n")
                cfg = dict(PRESETS["none"])
                save(cfg)
                return cfg
        cfg["api_key"] = key
        model = input(f"  Model [{cfg['model']}]: ").strip()
        if model:
            cfg["model"] = model
    else:
        cfg = dict(PRESETS["ollama"])
        url = input(f"  Ollama URL [{cfg['base_url']}]: ").strip()
        if url:
            cfg["base_url"] = url.rstrip("/")
        model = input(f"  Model [{cfg['model']}]: ").strip()
        if model:
            cfg["model"] = model

    save(cfg)
    print(f"\n  Saved: {describe(cfg)}")
    print(f"  Stored in {CONFIG_PATH} (chmod 600, gitignored)")
    print("  Change it any time with: ./run.sh --setup\n")
    return cfg


def resolve(interactive: bool = True) -> dict:
    """Load config, running the wizard on first use if we are on a terminal."""
    cfg = load()
    if cfg is not None:
        return cfg
    if interactive and os.isatty(0):
        return wizard()
    return dict(PRESETS["none"])
