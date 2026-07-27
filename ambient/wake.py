"""
Wake word detection (spec 4.3).

openWakeWord runs on CPU and the mic audio is never streamed anywhere -- the
model is local. Until the wake word fires we do no STT at all, which is both a
privacy property and a CPU saving.

If openwakeword is unavailable the detector degrades to "always awake" so the
rest of the loop stays testable. That degradation is logged loudly.
"""

from __future__ import annotations

import numpy as np

import config
from ambient.state import log_event


class AlwaysAwake:
    """Fallback: every frame counts as a wake. Dev only."""

    name = "always-awake"
    available = False

    def detect(self, _frame: np.ndarray) -> bool:
        return False  # the loop uses VAD directly in this mode

    def reset(self) -> None:
        pass


class OpenWakeWord:
    name = "openwakeword"
    available = True

    def __init__(self, model: str, threshold: float) -> None:
        from openwakeword.model import Model

        self.threshold = threshold
        self.model_name = model
        self._model = Model(wakeword_models=[model], inference_framework="onnx")
        # openWakeWord expects 80 ms (1280 samples) chunks at 16 kHz; we get
        # 32 ms frames, so we buffer up to a multiple of 1280.
        self._buffer = np.zeros(0, dtype=np.int16)
        self._chunk = 1280

    def detect(self, frame: np.ndarray) -> bool:
        self._buffer = np.concatenate([self._buffer, frame])
        fired = False
        while len(self._buffer) >= self._chunk:
            chunk = self._buffer[: self._chunk]
            self._buffer = self._buffer[self._chunk:]
            scores = self._model.predict(chunk)
            score = max(scores.values()) if scores else 0.0
            if score >= self.threshold:
                fired = True
                self.reset()
                log_event("wake_word", model=self.model_name,
                          score=round(float(score), 3))
                break
        return fired

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.int16)
        try:
            self._model.reset()
        except Exception:
            pass


def load_wake_word():
    if not config.WAKE_ENABLED:
        print("[wake] Wake word disabled -- listening continuously (dev mode).")
        return AlwaysAwake()
    try:
        detector = OpenWakeWord(config.WAKE_MODEL, config.WAKE_THRESHOLD)
        log_event("wake_loaded", model=config.WAKE_MODEL)
        print(f"[wake] Listening for '{config.WAKE_MODEL}'.")
        return detector
    except Exception as exc:
        log_event("wake_fallback", reason=str(exc)[:200])
        print(f"[wake] openWakeWord unavailable ({exc}).")
        print("[wake] Falling back to continuous listening -- dev mode only.")
        return AlwaysAwake()
