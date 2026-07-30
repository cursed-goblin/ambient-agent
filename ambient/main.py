"""
Orchestrator -- the ambient loop, plus AI escalation and the local UI.

    wake word -> AEC'd mic -> VAD -> Whisper -> rules -> Piper
                                   ^              |
                                   |              +-- no match? -> AI model
                                   +---- barge-in interrupt ------+

The rules layer is still the only thing that can touch the operating system.
The AI model can only produce words. That separation is the whole safety story:
a model that cannot act cannot act wrongly.

While we are speaking, the main loop KEEPS READING MIC FRAMES and feeds them to
a barge-in detector. Speaking happens on a worker thread. That is what makes
interruption feel like "Hey Google" rather than a school project.
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

import config
from ambient import (
    actions,
    llm,
    provider,
    rules,
    stt,
    tts,
    ui as ui_mod,
    vad as vad_mod,
    wake as wake_mod,
)
from ambient.audio_io import AudioUnavailable, MicStream, Speaker, list_devices
from ambient.state import Activity, Mode, current, log_event, timer

STATE = current()


class Assistant:
    def __init__(self, text_only: bool = False, with_ui: bool = False,
                 no_audio: bool = False) -> None:
        self.text_only = text_only
        self.with_ui = with_ui
        self.no_audio = no_audio
        self.running = True

        # --- brains that need no audio ---------------------------------
        self.timers = actions.TimerService(on_fire=self._announce)
        self.dispatch = actions.build_dispatch(self.timers)
        self.escalator: Optional[llm.Escalator] = None
        self._current_cfg: dict = {}

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

        # --- ui --------------------------------------------------------
        self.ui: Optional[ui_mod.UiServer] = None
        self._ui_queue: "queue.Queue[str]" = queue.Queue()

        self.awake = False
        self.awake_until = 0.0
        self._speech_thread: Optional[threading.Thread] = None
        self._handle_lock = threading.Lock()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def load(self, interactive_setup: bool = True) -> None:
        config.ensure_dirs()

        # Which model handles what the rules cannot?
        # With --ui, skip the terminal wizard -- the browser handles setup.
        cfg = provider.resolve(interactive=interactive_setup and not self.with_ui)
        self._current_cfg = cfg
        self.escalator = llm.load_escalator(cfg)
        print(f"[ai] {provider.describe(cfg)}")
        if self.escalator.enabled and not cfg.get("private"):
            print("[ai] Cloud model: the text of escalated requests leaves this")
            print("[ai] machine. Audio never does.")

        voice = tts.load_voice()

        if self.with_ui:
            self.ui = ui_mod.UiServer(
                on_command=self._ui_queue.put,
                get_config=lambda: self._current_cfg,
                on_config_change=self._on_ui_config_change,
            )
            url = self.ui.start()
            print(f"[ui] Serving on {url}")
            has_provider = cfg.get("provider") not in (None, "none", "")
            if has_provider:
                self.ui.add_message("system", f"Connected. AI: {provider.describe(cfg)}")
            else:
                self.ui.add_message("system",
                    "Welcome! Use the setup wizard to connect a Groq API key or local Ollama.")
            threading.Thread(target=self._ui_worker, daemon=True).start()

        if self.text_only:
            self.speech = tts.Speech(voice, speaker=None)
            self.transcriber = None
            print("[mode] Text-only. Type commands; no mic, no speaker.")
            return

        if self.no_audio:
            self.speech = tts.Speech(voice, speaker=None)
            self.transcriber = None
            print("[mode] No audio. Drive it from the UI.")
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
    # ui plumbing
    # ------------------------------------------------------------------
    def _ui_worker(self) -> None:
        """Typed commands from the browser, handled on our own thread."""
        while self.running:
            try:
                text = self._ui_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.handle(text)
            self._await_speech()

    def _ui_message(self, role: str, text: str) -> None:
        if self.ui is not None:
            self.ui.add_message(role, text)

    def _on_ui_config_change(self, cfg: dict) -> None:
        """Hot-reload the escalator when the user saves new settings in the UI."""
        self._current_cfg = cfg
        self.escalator = llm.load_escalator(cfg)
        desc = provider.describe(cfg)
        print(f"[ai] provider reloaded: {desc}")
        if self.ui:
            self.ui.add_message("system", f"AI provider updated: {desc}")

    # ------------------------------------------------------------------
    # speaking (worker thread) + barge-in
    # ------------------------------------------------------------------
    def _speak_async(self, text: str) -> None:
        STATE.set_activity(Activity.SPEAKING, text[:60])
        STATE.last_reply = text
        self._ui_message("agent", text)

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

    def _escalate(self, text: str) -> Optional[str]:
        """Ask the model. Returns a reply, or None meaning 'refuse'."""
        if self.escalator is None or not self.escalator.enabled:
            return None
        STATE.set_activity(Activity.THINKING, "Thinking...")
        return self.escalator.answer(text)

    def handle(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        with self._handle_lock:
            print(f"\n  > {text}")
            self._ui_message("you", text)
            log_event("heard", text=text)
            STATE.set_activity(Activity.THINKING, "Matching command...")

            clock = timer("route")
            intent = rules.match(text)
            clock.stop(intent=intent.name if intent else None)

            # ---- CONTROL: never gated, never routed to a model --------
            if rules.is_control(intent):
                self._handle_control(intent)
                return

            if intent is None:
                reply = self._escalate(text)
                if reply is None:
                    log_event("refused", text=text, reason="no_rule_match")
                    self._speak_async(config.REFUSAL_LINE)
                else:
                    self._speak_async(reply)
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
            self._ui_message("system", "stopped")
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

    def run_idle(self) -> None:
        """No mic, no stdin -- just the UI. Used by --ui --no-audio."""
        print("\n[ready] UI only. Ctrl-C to quit.\n")
        while self.running:
            time.sleep(0.5)

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
        elif self.mic is None:
            self.run_idle()
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
            if self.ui is not None:
                self.ui.stop()
        finally:
            log_event("shutdown")


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------

def _check_ai() -> int:
    """Verify the configured provider actually answers."""
    cfg = provider.resolve(interactive=True)
    print(f"\nProvider: {provider.describe(cfg)}")
    if cfg.get("provider") in ("none", "", None):
        print("Nothing to check -- rules only.\n")
        return 0
    client = llm.LlmClient(cfg["base_url"], cfg.get("api_key", ""), cfg["model"])
    print(f"Calling {cfg['base_url']} ...")
    ok, detail = client.ping()
    print(("  OK: " if ok else "  FAILED: ") + detail + "\n")
    if not ok and cfg["provider"] == "ollama":
        print("Is Ollama running?  ollama serve")
        print(f"Is the model pulled?  ollama pull {cfg['model']}\n")
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ambient assistant -- voice loop, rules, AI escalation, UI"
    )
    parser.add_argument("--text", action="store_true",
                        help="type commands instead of speaking (no audio needed)")
    parser.add_argument("--ui", action="store_true",
                        help="serve the local web UI at http://127.0.0.1:8765")
    parser.add_argument("--no-audio", action="store_true",
                        help="skip the mic entirely; drive it from the UI only")
    parser.add_argument("--setup", action="store_true",
                        help="choose the AI provider (Groq API or local Ollama) via terminal")
    parser.add_argument("--check-ai", action="store_true",
                        help="send one test request to the configured provider")
    parser.add_argument("--devices", action="store_true",
                        help="list audio devices and exit")
    parser.add_argument("--say", metavar="TEXT",
                        help="speak one line and exit (TTS smoke test)")
    args = parser.parse_args(argv)

    if args.devices:
        print(list_devices())
        return 0

    if args.setup:
        provider.wizard()
        return 0

    if args.check_ai:
        return _check_ai()

    text_only = args.text or (args.no_audio and not args.ui)
    assistant = Assistant(
        text_only=text_only,
        with_ui=args.ui,
        no_audio=args.no_audio and not text_only,
    )
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
