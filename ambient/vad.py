"""
Voice activity detection (spec 4.3).

Silero VAD, 32 ms frames, running CONTINUOUSLY -- including while our own TTS
is playing. Speech detected during playback is the barge-in trigger.

Falls back to a simple RMS energy gate if torch/silero is unavailable, so the
rest of the pipeline stays testable on a machine without the model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import config
from ambient.state import log_event


class EnergyVad:
    """Crude fallback. Good enough to smoke-test the loop, not to ship."""

    name = "energy"

    def __init__(self, rms_threshold: float = 550.0) -> None:
        self.rms_threshold = rms_threshold

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))
        return min(1.0, rms / (self.rms_threshold * 2))

    def reset(self) -> None:
        pass


class SileroVad:
    name = "silero"

    def __init__(self) -> None:
        import torch  # imported lazily so the fallback path needs no torch

        self._torch = torch
        torch.set_num_threads(1)
        self._model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=False,
            trust_repo=True,
            verbose=False,
        )
        self._model.eval()

    def probability(self, frame: np.ndarray) -> float:
        audio = frame.astype(np.float32) / 32768.0
        tensor = self._torch.from_numpy(audio)
        with self._torch.no_grad():
            return float(self._model(tensor, config.SAMPLE_RATE).item())

    def reset(self) -> None:
        try:
            self._model.reset_states()
        except Exception:
            pass


def load_vad():
    try:
        vad = SileroVad()
        log_event("vad_loaded", backend="silero")
        return vad
    except Exception as exc:
        log_event("vad_fallback", backend="energy", reason=str(exc)[:200])
        print(f"[vad] Silero unavailable ({exc}); using energy fallback.")
        return EnergyVad()


class UtteranceDetector:
    """
    Turns a stream of VAD probabilities into utterance boundaries.

    start : SPEECH_START_FRAMES consecutive speech frames
    end   : SPEECH_END_FRAMES consecutive silence frames
    """

    def __init__(self, vad, threshold: Optional[float] = None) -> None:
        self.vad = vad
        self.threshold = config.VAD_THRESHOLD if threshold is None else threshold
        self._speech_run = 0
        self._silence_run = 0
        self.active = False

    def reset(self) -> None:
        self._speech_run = 0
        self._silence_run = 0
        self.active = False
        self.vad.reset()

    def push(self, frame: np.ndarray) -> str:
        """Returns one of: '', 'start', 'speech', 'end'."""
        prob = self.vad.probability(frame)
        is_speech = prob >= self.threshold

        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0

        if not self.active:
            if self._speech_run >= config.SPEECH_START_FRAMES:
                self.active = True
                return "start"
            return ""

        if self._silence_run >= config.SPEECH_END_FRAMES:
            self.active = False
            return "end"
        return "speech"


class BargeInDetector:
    """
    Separate, stricter detector used only while we are speaking.

    A higher threshold means residual echo that survives AEC cannot make the
    assistant interrupt itself.
    """

    def __init__(self, vad) -> None:
        self.vad = vad
        self._run = 0

    def reset(self) -> None:
        self._run = 0

    def push(self, frame: np.ndarray) -> bool:
        prob = self.vad.probability(frame)
        if prob >= config.BARGE_IN_VAD_THRESHOLD:
            self._run += 1
        else:
            self._run = 0
        return self._run >= config.BARGE_IN_FRAMES
