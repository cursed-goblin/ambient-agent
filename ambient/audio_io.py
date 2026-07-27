"""
Audio capture and playback (spec 4.3).

Full duplex. The mic stream stays open while TTS plays -- that is what makes
barge-in possible. We rely on PipeWire's echo-cancel module (see
setup_audio.sh) to remove our own speaker output from the mic signal.

We deliberately do NOT mute the mic during playback.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator, Optional

import numpy as np

import config
from ambient.state import log_event

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - missing lib or no audio backend
    sd = None


class AudioUnavailable(RuntimeError):
    pass


def require_sounddevice():
    if sd is None:
        raise AudioUnavailable(
            "sounddevice is not available. Install it with\n"
            "    pip install sounddevice\n"
            "and make sure libportaudio2 is installed."
        )
    return sd


def list_devices() -> str:
    if sd is None:
        return "sounddevice unavailable"
    return str(sd.query_devices())


def _resolve(device, kind: str):
    """Accept a substring of a device name, or None for the system default."""
    if device is None or sd is None:
        return None
    if isinstance(device, int):
        return device
    needle = str(device).lower()
    for idx, info in enumerate(sd.query_devices()):
        channels = info["max_input_channels"] if kind == "input" else info["max_output_channels"]
        if channels > 0 and needle in info["name"].lower():
            return idx
    log_event("audio_device_not_found", requested=device, kind=kind)
    return None


class MicStream:
    """
    Continuous 16 kHz mono int16 capture, exposed as an iterator of frames.

    One frame == config.FRAME_SAMPLES (32 ms) so it feeds Silero VAD directly.
    A bounded queue means that if a consumer stalls we drop the oldest audio
    rather than growing memory forever.
    """

    def __init__(self, maxsize: int = 200) -> None:
        require_sounddevice()
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=maxsize)
        self._stream: Optional[object] = None
        self._dropped = 0

    def _callback(self, indata, _frames, _time, status) -> None:
        if status:
            log_event("audio_input_status", status=str(status))
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self._dropped += 1
            try:
                self._q.get_nowait()
                self._q.put_nowait(indata[:, 0].copy())
            except queue.Empty:
                pass

    def start(self) -> "MicStream":
        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=config.FRAME_SAMPLES,
            channels=config.CHANNELS,
            dtype=config.DTYPE,
            device=_resolve(config.INPUT_DEVICE, "input"),
            callback=self._callback,
        )
        self._stream.start()
        log_event("mic_started", rate=config.SAMPLE_RATE,
                  frame_ms=config.FRAME_MS)
        return self

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            yield self._q.get()

    def drain(self) -> None:
        """Discard buffered audio -- used after we finish speaking."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log_event("mic_stopped", dropped_frames=self._dropped)

    def __enter__(self) -> "MicStream":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


class Speaker:
    """
    Chunked, interruptible playback.

    Audio is pushed in small chunks (config.TTS_CHUNK_MS). `stop()` clears the
    queue and aborts the stream immediately, which is the mechanical half of
    barge-in. Never render one long buffer -- you cannot cut it cleanly.
    """

    def __init__(self, sample_rate: Optional[int] = None) -> None:
        require_sounddevice()
        self.sample_rate = sample_rate or config.PIPER_SAMPLE_RATE
        self._q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._stream = None
        self._worker: Optional[threading.Thread] = None
        self._playing = threading.Event()
        self._abort = threading.Event()

    # -- lifecycle ------------------------------------------------------
    def _open(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=_resolve(config.OUTPUT_DEVICE, "output"),
        )
        self._stream.start()

    def _run(self) -> None:
        self._open()
        while not self._abort.is_set():
            chunk = self._q.get()
            if chunk is None:
                break
            if self._abort.is_set():
                break
            try:
                self._stream.write(chunk)
            except Exception as exc:  # pragma: no cover
                log_event("playback_error", error=str(exc))
                break
        self._playing.clear()

    # -- public API -----------------------------------------------------
    def begin(self) -> None:
        self._abort.clear()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._playing.set()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def feed(self, pcm: np.ndarray) -> None:
        """Queue int16 mono PCM, split into TTS_CHUNK_MS pieces."""
        if self._abort.is_set():
            return
        chunk_len = max(1, self.sample_rate * config.TTS_CHUNK_MS // 1000)
        for start in range(0, len(pcm), chunk_len):
            if self._abort.is_set():
                return
            self._q.put(pcm[start:start + chunk_len])

    def end(self) -> None:
        """Signal end of utterance and wait for the queue to flush."""
        self._q.put(None)
        if self._worker is not None:
            self._worker.join(timeout=30)
        self._playing.clear()

    def stop(self) -> None:
        """Immediate abort -- this is the barge-in cut."""
        self._abort.set()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._q.put(None)
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:
                pass
        self._playing.clear()

    @property
    def playing(self) -> bool:
        return self._playing.is_set()

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
