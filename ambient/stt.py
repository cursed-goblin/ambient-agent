"""
Speech to text (spec 4.4).

faster-whisper on CPU with int8. `base.en` is the sensible starting point --
no GPU required. Two behaviours matter here:

1. Domain vocabulary biasing via initial_prompt. This materially improves
   recognition of "Flipkart", "UPI", "brightness" etc.
2. Partial transcripts. Streaming partials to the overlay is the single
   biggest perceived-quality win available, because users forgive latency
   when they can see they were heard.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

import config
from ambient.state import log_event, timer


class Transcriber:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        load = timer("whisper_load")
        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
        )
        load.stop(model=config.WHISPER_MODEL)
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: int16 mono @ 16 kHz. Returns stripped text ('' if silence)."""
        if audio.size == 0:
            return ""
        samples = audio.astype(np.float32) / 32768.0
        clock = timer("stt")
        with self._lock:
            segments, _info = self.model.transcribe(
                samples,
                language="en",
                beam_size=config.WHISPER_BEAM,
                vad_filter=False,             # we already gated with Silero
                initial_prompt=config.WHISPER_PROMPT,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text for seg in segments).strip()
        clock.stop(chars=len(text), audio_s=round(len(audio) / config.SAMPLE_RATE, 2))
        return text


class PartialTranscriber:
    """
    Runs the transcriber on a growing buffer in a background thread while the
    user is still speaking, so the overlay can show live text.

    Partials are best-effort: if the machine cannot keep up we simply skip a
    round rather than delaying the final result.
    """

    def __init__(self, transcriber: Transcriber,
                 on_partial: Callable[[str], None]) -> None:
        self.transcriber = transcriber
        self.on_partial = on_partial
        self._busy = threading.Lock()
        self._last_run = 0.0

    def maybe_run(self, audio: np.ndarray) -> None:
        if not config.EMIT_PARTIALS:
            return
        now = time.monotonic()
        if now - self._last_run < config.PARTIAL_INTERVAL_S:
            return
        if not self._busy.acquire(blocking=False):
            return
        self._last_run = now
        snapshot = audio.copy()

        def worker() -> None:
            try:
                text = self.transcriber.transcribe(snapshot)
                if text:
                    self.on_partial(text)
            except Exception as exc:
                log_event("partial_error", error=str(exc)[:200])
            finally:
                self._busy.release()

        threading.Thread(target=worker, daemon=True).start()


def load_transcriber() -> Optional[Transcriber]:
    try:
        return Transcriber()
    except Exception as exc:
        log_event("stt_unavailable", reason=str(exc)[:200])
        print(f"[stt] faster-whisper unavailable: {exc}")
        return None
