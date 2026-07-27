"""
Audio sanity checks -- run this BEFORE the main loop.

It answers three questions, in order of importance:

1. Can we open a 16 kHz mono input stream at all?
2. Is echo cancellation actually working? (play a tone, measure mic level)
3. Does the VAD see your voice at a sane level above the noise floor?

Usage:
    python3 tools/check_audio.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from ambient.audio_io import MicStream, Speaker, list_devices


def rms(frame):
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))


def measure(mic, seconds):
    frames = int(seconds * 1000 / config.FRAME_MS)
    values = []
    stream = mic.frames()
    for _ in range(frames):
        values.append(rms(next(stream)))
    return sum(values) / max(1, len(values))


def tone(seconds, rate, freq=440.0):
    t = np.arange(int(seconds * rate)) / rate
    wave = 0.35 * np.sin(2 * np.pi * freq * t)
    return (wave * 32767).astype(np.int16)


def main():
    print("=" * 62)
    print("AUDIO CHECK")
    print("=" * 62)

    print("\n-- devices --")
    print(list_devices())
    print("\ninput  device filter : %r" % (config.INPUT_DEVICE,))
    print("output device filter : %r" % (config.OUTPUT_DEVICE,))

    if config.INPUT_DEVICE is None:
        print("\nNOTE: no input filter set. If you ran setup_audio.sh, use:")
        print("      export AMBIENT_INPUT_DEVICE=echo_cancel")

    print("\n[1/3] Opening mic ...")
    try:
        mic = MicStream().start()
    except Exception as exc:
        print("      FAILED: %s" % exc)
        return 1
    print("      ok")

    print("\n[2/3] Measuring room noise. Stay quiet for 2 seconds ...")
    time.sleep(0.3)
    mic.drain()
    floor = measure(mic, 2.0)
    print("      noise floor RMS = %.0f" % floor)
    if floor > 900:
        print("      WARNING: noisy room. Expect false VAD triggers.")

    print("\n[3/3] Echo cancellation test. Playing a tone -- stay quiet ...")
    leak = None
    try:
        speaker = Speaker(sample_rate=config.SAMPLE_RATE)
        speaker.begin()
        speaker.feed(tone(2.0, config.SAMPLE_RATE))
        mic.drain()
        time.sleep(0.4)
        leak = measure(mic, 1.2)
        speaker.end()
        speaker.close()
    except Exception as exc:
        print("      could not play tone: %s" % exc)

    if leak is not None:
        ratio = leak / max(1.0, floor)
        print("      mic RMS while speaking = %.0f  (%.1fx floor)" % (leak, ratio))
        if ratio < 2.0:
            print("      EXCELLENT -- AEC is working. Barge-in will be reliable.")
        elif ratio < 5.0:
            print("      OK -- some leakage. Raise the barge-in threshold to 0.8:")
            print("           export AMBIENT_BARGE_IN_VAD_THRESHOLD=0.8")
        else:
            print("      BAD -- the mic clearly hears the speaker.")
            print("           The assistant will interrupt itself mid-sentence.")
            print("           Fix: run ./setup_audio.sh, or use headphones,")
            print("           or set AMBIENT_INPUT_DEVICE=echo_cancel")

    print("\n[bonus] Say something for 3 seconds ...")
    mic.drain()
    voice = measure(mic, 3.0)
    print("        voice RMS = %.0f  (%.1fx floor)" % (voice, voice / max(1.0, floor)))
    if voice < floor * 3:
        print("        WARNING: voice barely above noise. Move closer.")
    else:
        print("        Good separation.")

    mic.stop()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
