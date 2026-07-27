"""
Phase 1 orchestrator -- the ambient voice loop.

    wake word -> AEC'd mic -> VAD -> Whisper -> rules layer -> Piper
                                   ^                              |
                                   +------ barge-in interrupt ----+

No LLM. No network. No GPU required.

The single most important property in this file: while we are speaking, the
main loop KEEPS READING MIC FRAMES and feeds them to a barge-in detector.
Speaking happens on a worker thread. That is what makes interruption feel
like "Hey Google" instead of like a school project.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

import config
from ambient import actions, rules, stt, tts, vad as vad_mod, wake as wake_mod
from ambient.audio_io import AudioUnavailable, MicStream, Speaker, list_devices
from ambient.state import Activity, Mode, current, log_event, timer

STATE = current()


class Assistant:
    def __init__(self, text_only: bool = False) -> None:
        self.text_only = text_only
        self.running = True

        # --- brains that need no audio ---------------------------------
        self.timers = actions.TimerService(on_fire=self._announce)
        self.dispatch = actions.build_dispatch(self.timers)

        # --- audio stack ----------------------------------------------
        self.vad = None
        self.utterance = None
        self.barge_in = None
        self.wake = None
        self.transcriber = None
        self.partials = None
        self.speech = None
        self.mic: Optional[MicStream] = None
        self.speaker: Optional[Speaker] = None

        self.awake = False
        self.awake_until = 0.0
        self._speech_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def load(self) -> None:
        config.ensure_dirs()
        voice = tts.load_voice()

        if self.text_only:
            self.speech = tts.Speech(voice, speaker=None)
            self.transcriber = None
            print("[mode] Text-only. Type commands; no mic, no speaker.")
            return

        self.vad = vad_mod.load_vad()
        self.utterance = vad_mod.UtteranceDetector(self.vad)
        self.barge_in = vad_mod.BargeInDetector(self.vad)
        self.wake = wake_mod.load_wake_word()
        self.transcriber = stt.load_transcriber()
        if self.transcriber is not None:
            self.partials = stt.PartialTranscriber(self.transcriber, self._on_partial)

        try:
            self.speaker = Speaker(sample_rate=getattr(voice, "sample_rate",
                                                       config.SAMPLE_RATE))
            self.mic = MicStream().start()
        except AudioUnavailable as exc:
            print(f"[audio] {exc}")
            print("[audio] Run with --text to work without audio devices.")
            raise SystemExit(2)

        self.speech = tts.Speech(voice, self.speaker)

    # ------------------------------------------------------------------
    # speaking (worker thread) + barge-in
    # ------------------------------------------------------------------
    def _speak_async(self, text: str) -> None:
        STATE.set_activity(Activity.SPEAKING, text[:60])
        STATE.last_reply = text

        def worker() -> None:
            completed = self.speech.speak(text)
            log_event("reply", text=text[:200], completed=completed)
            STATE.set_activity(Activity.IDLE)
            # Keep the mic open briefly for a natural follow-up.
            self.awake_until = time.monotonic() + config.FOLLOW_UP_WINDOW_S
            if self.mic is not None and completed:
                self.mic.drain()

        self._speech_thread = threading.Thread(target=worker, daemon=True)
        self._speech_thread.start()
        if self.barge_in is not None:
            self.barge_in.reset()

    def _announce(self, text: str) -> None:
        """Used by the timer service firing from its own thread."""
        if self.speech is not None:
            self._speak_async(text)

    def _await_speech(self) -> None:
        if self._speech_thread is not None:
            self._speech_thread.join(timeout=60)
            self._speech_thread = None

    # ------------------------------------------------------------------
    # transcript handling
    # ------------------------------------------------------------------
    def _on_partial(self, text: str) -> None:
        STATE.set_partial(text)
        sys.stdout.write(f"\r  ... {text[:70]:<72}")
        sys.stdout.flush()

    def handle(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        print(f"\n  > {text}")
        log_event("heard", text=text)
        STATE.set_activity(Activity.THINKING, "Matching command...")

        clock = timer("route")
        intent = rules.match(text)
        clock.stop(intent=intent.name if intent else None)

        # ---- CONTROL: never gated, never routed to a model ------------
        if rules.is_control(intent):
            self._handle_control(intent)
            return

        if intent is None:
            # Reject by default (spec 4.6). Phase 4 escalates here instead.
            log_event("refused", text=text, reason="no_rule_match")
            self._speak_async(config.REFUSAL_LINE)
            return

        handler = self.dispatch.get(intent.name)
        if handler is None:
            log_event("refused", text=text, reason="no_handler",
                      intent=intent.name)
            self._speak_async(config.REFUSAL_LINE)
            return

        STATE.set_activity(Activity.ACTING, intent.name)
        log_event("intent", name=intent.name, slots=intent.slots,
                  risk=intent.risk)
        try:
            reply = handler(intent)
        except Exception as exc:
            log_event("action_error", intent=intent.name, error=str(exc)[:200])
            reply = "That didn't work."
        self._speak_async(reply)

    def _handle_control(self, intent: rules.Intent) -> None:
        name = intent.name
        log_event("control", name=name)

        if name == "control.stop":
            self.speech.interrupt()
            self.timers.cancel(silent=True)
            STATE.set_activity(Activity.IDLE)
            print("  [stopped]")
            return

        if name == "mode.manual":
            self.speech.interrupt()
            STATE.set_mode(Mode.MANUAL)
            self.awake = False
            self._speak_async("Going quiet. Say the wake word when you need me.")
            return

        if name == "mode.assist":
            STATE.set_mode(Mode.ASSIST)
            self._speak_async("Assist mode. Camera on.")
            return

        if name == "mode.ambient":
            STATE.set_mode(Mode.AMBIENT)
            self._speak_async("Listening.")
            return

    # ------------------------------------------------------------------
    # main loops
    # ------------------------------------------------------------------
    def run_text(self) -> None:
        print("\nType a command ('quit' to exit).\n")
        while self.running:
            try:
                line = input("you > ")
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip().lower() in ("quit", "exit"):
                break
            self.handle(line)
            self._await_speech()

    def run_voice(self) -> None:
        buffer: list[np.ndarray] = []
        started_at = 0.0
        print("\n[ready] Ambient loop running. Ctrl-C to quit.\n")

        for frame in self.mic.frames():
            if not self.running:
                break

            # --- manual mode: consume and discard -----------------------
            if STATE.mode is Mode.MANUAL:
                continue

            # --- barge-in: we are talking, is the user talking over us? -
            if self.speech.speaking.is_set():
                if self.barge_in.push(frame):
                    clock = timer("barge_in")
                    self.speech.interrupt()
                    clock.stop()
                    print("\n  [interrupted]")
                    self.barge_in.reset()
                    self.utterance.reset()
                    buffer = [frame]
                    started_at = time.monotonic()
                    self.awake = True
                    self.awake_until = time.monotonic() + config.FOLLOW_UP_WINDOW_S
                    STATE.set_activity(Activity.LISTENING)
                continue

            # --- wake gate ---------------------------------------------
            if not self.awake:
                if getattr(self.wake, "available", False):
                    if self.wake.detect(frame):
                        self.awake = True
                        self.awake_until = time.monotonic() + config.FOLLOW_UP_WINDOW_S
                        self.utterance.reset()
                        buffer = []
                        STATE.set_activity(Activity.WAKE)
                        print("  [wake]")
                    continue
                # No wake model: stay awake (dev mode).
                self.awake = True
                self.awake_until = time.monotonic() + 1e9

            if time.monotonic() > self.awake_until and not self.utterance.active:
                self.awake = False
                STATE.set_activity(Activity.IDLE)
                STATE.set_partial("")
                continue

            # --- utterance capture -------------------------------------
            event = self.utterance.push(frame)

            if event == "start":
                buffer = [frame]
                started_at = time.monotonic()
                STATE.set_activity(Activity.LISTENING)

            elif event == "speech":
                buffer.append(frame)
                if self.partials is not None and len(buffer) % 8 == 0:
                    self.partials.maybe_run(np.concatenate(buffer))
                if time.monotonic() - started_at > config.MAX_UTTERANCE_S:
                    event = "end"

            if event == "end" and buffer:
                audio = np.concatenate(buffer)
                buffer = []
                self.utterance.reset()
                STATE.set_partial("")
                if self.transcriber is None:
                    print("\n  [stt unavailable -- cannot transcribe]")
                    continue
                text = self.transcriber.transcribe(audio)
                self.handle(text)

    def run(self) -> None:
        if self.text_only:
            self.run_text()
        else:
            self.run_voice()

    def shutdown(self) -> None:
        self.running = False
        try:
            if self.speech is not None:
                self.speech.interrupt()
            self.timers.cancel(silent=True)
            if self.mic is not None:
                self.mic.stop()
            if self.speaker is not None:
                self.speaker.close()
        finally:
            log_event("shutdown")


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ambient assistant -- Phase 1 voice loop"
    )
    parser.add_argument("--text", action="store_true",
                        help="type commands instead of speaking (no audio needed)")
    parser.add_argument("--devices", action="store_true",
                        help="list audio devices and exit")
    parser.add_argument("--say", metavar="TEXT",
                        help="speak one line and exit (TTS smoke test)")
    args = parser.parse_args(argv)

    if args.devices:
        print(list_devices())
        return 0

    assistant = Assistant(text_only=args.text)
    assistant.load()

    if args.say:
        assistant._speak_async(args.say)
        assistant._await_speech()
        assistant.shutdown()
        return 0

    def on_signal(_sig, _frame):
        print("\n[shutdown]")
        assistant.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        assistant.run()
    finally:
        assistant.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
