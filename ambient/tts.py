"""
Text to speech (spec 4.5).

Piper, streamed as raw PCM through a pipe, fed to the speaker in ~200 ms
chunks. This is the shape that makes barge-in possible: we never render one
long WAV, so we can cut cleanly at any moment.

Piper is real-time on CPU, which is why it is the production choice.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from typing import Callable, Optional

import numpy as np

import config
from ambient.state import log_event, timer


class PiperUnavailable(RuntimeError):
    pass


class Piper:
    """Thin wrapper around the piper CLI producing int16 PCM on stdout."""

    def __init__(self) -> None:
        if not shutil.which(config.PIPER_BIN):
            raise PiperUnavailable(f"'{config.PIPER_BIN}' not found on PATH")
        if not os.path.exists(config.PIPER_VOICE):
            raise PiperUnavailable(f"voice model missing: {config.PIPER_VOICE}")
        self.sample_rate = config.PIPER_SAMPLE_RATE

    def stream(
        self,
        text: str,
        on_chunk: Callable[[np.ndarray], None],
        should_stop: Callable[[], bool],
    ) -> bool:
        """
        Synthesise `text`, handing PCM chunks to `on_chunk` as they arrive.

        Returns True if playback completed, False if it was interrupted.
        Checks `should_stop()` between every chunk -- that is the barge-in hook.
        """
        cmd = [
            config.PIPER_BIN,
            "--model", config.PIPER_VOICE,
            "--output_raw",
        ]
        clock = timer("tts_first_chunk")
        first = True
        # 200 ms of int16 mono at the voice's rate
        read_size = max(1024, self.sample_rate * config.TTS_CHUNK_MS // 1000 * 2)

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()
            while True:
                if should_stop():
                    proc.kill()
                    log_event("tts_interrupted", text=text[:80])
                    return False
                raw = proc.stdout.read(read_size)
                if not raw:
                    break
                if first:
                    clock.stop(chars=len(text))
                    first = False
                on_chunk(np.frombuffer(raw, dtype=np.int16))
            return True
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.wait(timeout=5)


class ConsoleVoice:
    """
    Fallback "voice" that prints instead of speaking.

    Keeps the whole loop runnable on a machine with no Piper model, which is
    how you develop the rules layer before the audio stack is set up.
    """

    sample_rate = config.SAMPLE_RATE

    def stream(self, text, on_chunk, should_stop) -> bool:  # noqa: ARG002
        print(f"\n  [speak] {text}")
        return True


def load_voice():
    if not config.TTS_ENABLED:
        return ConsoleVoice()
    try:
        voice = Piper()
        log_event("tts_loaded", voice=config.PIPER_VOICE)
        return voice
    except Exception as exc:
        log_event("tts_fallback", reason=str(exc)[:200])
        print(f"[tts] Piper unavailable: {exc}")
        print("[tts] Falling back to printed replies.")
        return ConsoleVoice()


class Speech:
    """
    Owns "say this, and let the user cut me off".

    speak() blocks until the utterance finishes or is interrupted. interrupt()
    is safe to call from any thread -- it is what the barge-in detector calls.
    """

    def __init__(self, voice, speaker) -> None:
        self.voice = voice
        self.speaker = speaker
        self._interrupt = threading.Event()
        self.last_text = ""
        self.speaking = threading.Event()

    def interrupt(self) -> None:
        if self.speaking.is_set():
            self._interrupt.set()
            if self.speaker is not None:
                self.speaker.stop()

    def speak(self, text: str) -> bool:
        """Returns True if fully spoken, False if interrupted."""
        text = (text or "").strip()
        if not text:
            return True
        self.last_text = text
        self._interrupt.clear()
        self.speaking.set()

        if self.speaker is None:
            completed = self.voice.stream(text, lambda _c: None,
                                          self._interrupt.is_set)
            self.speaking.clear()
            return completed

        self.speaker.begin()
        try:
            completed = self.voice.stream(
                text,
                self.speaker.feed,
                self._interrupt.is_set,
            )
            if completed:
                self.speaker.end()
            return completed
        finally:
            self.speaking.clear()
            self._interrupt.clear()
