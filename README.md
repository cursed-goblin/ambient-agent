# ambient-agent — Phase 1

The hands-free, local-first assistant from the spec. This is **Phase 1 only**:
the voice loop and the deterministic rules layer. No LLM, no browser
automation, no gestures, no network calls. Those are Phases 2–4.

Phase 1 is deliberately the boring part, because it is the part that decides
whether the whole idea feels good or feels broken. If wake → hear → act →
speak → interrupt is not solid, nothing built on top of it will be.

```
wake word ──► AEC'd mic ──► VAD ──► Whisper ──► rules ──► Piper
                             ▲                              │
                             └────── barge-in interrupt ◄────┘
```

**Runs on CPU. No GPU needed.** That matters because you do not have a GPU
machine yet, and none of this phase requires one.

---

## Honest status

| Part | State |
| --- | --- |
| Rules layer (`ambient/rules.py`) | **Tested.** 25 unit tests pass. |
| Config, state, audit log | Written, imports clean. |
| OS actions (`ambient/actions.py`) | Written, `DRY_RUN` safe by default. Needs a real Ubuntu box. |
| Audio capture / playback | Written, **never run against a real mic.** |
| Wake / VAD / STT / TTS | Written, **models never downloaded or run.** |
| Barge-in timing | Written. The `<300ms` target is **unverified.** |

The audio path was authored in an environment with no microphone, no speaker
and no audio server. Expect to spend your first session tuning it. That is
normal and the tuning knobs are all in one file.

---

## Install

Ubuntu 24.04, GNOME. Your keyboard and mouse keep working throughout — this
is a normal app, not an OS replacement.

```bash
# 1. system packages
sudo apt update
sudo apt install -y python3-venv python3-dev libportaudio2 \
    pipewire pipewire-pulse wireplumber pulseaudio-utils \
    wmctrl brightnessctl playerctl network-manager

# 2. python env
cd ambient-agent
python3 -m venv .venv
source .venv/bin/activate

# 3. torch CPU-only FIRST (otherwise pip drags in ~2.5GB of CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. piper binary + a voice
mkdir -p models/piper && cd models/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
cd ../..
```

`piper` itself: either `pip install piper-tts` (gives you a `piper` command) or
grab a release binary from `github.com/rhasspy/piper` and put it on your PATH.

After cloning, make the scripts executable: `chmod +x run.sh setup_audio.sh`.

---

## Run it, in this order

**Step 1 — prove the logic works, with no audio at all.**

```bash
python3 tests/test_rules.py     # should print 25/25 passed
./run.sh --text                 # type commands, read replies
```

Try: `set volume to 40`, `open browser`, `timer for 10 minutes`, `what's 12 * 8`,
`battery`, `stop`, and something nonsense like `book me a flight to tokyo` —
that last one must be refused. Refusal is a feature.

**Step 2 — echo cancellation. Do not skip this.**

```bash
./setup_audio.sh
export AMBIENT_INPUT_DEVICE=echo_cancel
export AMBIENT_OUTPUT_DEVICE=echo_cancel
python3 tools/check_audio.py
```

You want the tone test to report `EXCELLENT` or `OK`. If it says `BAD`, the mic
is hearing the speakers and the assistant will interrupt itself on every single
reply. Headphones are the instant fix; AEC is the real one.

**Step 3 — TTS alone.**

```bash
./run.sh --say "Volume set to forty percent."
```

**Step 4 — the full loop.**

```bash
./run.sh
```

Say `hey jarvis` (the default openWakeWord model), wait for `[wake]`, then give
a command. While it is replying, talk over it — you should see `[interrupted]`.

**Step 5 — let it actually touch the OS.**

`DRY_RUN` is **on** by default: every action is logged, nothing executes. Once
you trust what you see in `var/audit.log`:

```bash
AMBIENT_DRY_RUN=0 ./run.sh
```

---

## Tuning

Everything lives in `config.py`, and every value can be overridden by an
`AMBIENT_`-prefixed environment variable, so you can tune without editing code.

| Symptom | Knob | Try |
| --- | --- | --- |
| Interrupts itself while speaking | `AMBIENT_BARGE_IN_VAD_THRESHOLD` | `0.8`–`0.9` |
| Slow to notice you cutting in | `AMBIENT_BARGE_IN_FRAMES` | `2` |
| Cuts you off mid-sentence | `AMBIENT_SPEECH_END_FRAMES` | `30`–`40` |
| Waits too long after you stop | `AMBIENT_SPEECH_END_FRAMES` | `15`–`20` |
| Wake word fires at the TV | `AMBIENT_WAKE_THRESHOLD` | `0.7` |
| Wake word never fires | `AMBIENT_WAKE_THRESHOLD` | `0.4` |
| Mishears product/app names | `AMBIENT_WHISPER_MODEL` | `small.en` |
| Too slow on your laptop | `AMBIENT_WHISPER_MODEL` | `tiny.en` |

Measure before you tune. `var/audit.log` is JSONL and records latency for wake,
STT, routing, TTS first-chunk and barge-in:

```bash
tail -f var/audit.log | python3 -c "import sys,json;[print(json.loads(l)) for l in sys.stdin]"
```

Develop in the kitchen, not at the desk. Cooking noise, three metres away,
running tap — that is the real test, and it is where the defaults will fail.

---

## Layout

```
config.py              every tunable, all env-overridable
ambient/rules.py       ~40 regex intents + tier-0 math/conversion. Tested.
ambient/state.py       modes, activity, JSONL audit log, latency timers
ambient/actions.py     volume, brightness, apps, timers, system info
ambient/audio_io.py    full-duplex mic + chunked killable speaker
ambient/vad.py         Silero VAD + separate stricter barge-in detector
ambient/wake.py        openWakeWord
ambient/stt.py         faster-whisper + partial transcripts
ambient/tts.py         Piper streaming, cancellable mid-utterance
ambient/main.py        the orchestrator
tools/check_audio.py   AEC verification
tests/test_rules.py    runs anywhere, no models needed
```

---

## Design decisions worth knowing

**The mic never mutes.** Not while speaking, not while thinking. Muting is the
obvious way to avoid hearing yourself and it is exactly why most hobby
assistants cannot be interrupted. We keep it open and rely on AEC.

**Two separate VAD thresholds.** Normal listening uses `0.5`. Listening *while
speaking* uses `0.7`, because residual echo that survives AEC looks a little
bit like speech. One threshold cannot serve both jobs.

**Speaking happens on a worker thread.** The main loop keeps consuming mic
frames the entire time. If speaking blocked the loop, barge-in would be
impossible by construction.

**Unknown input is refused, not guessed.** `rules.match()` returns `None` and
the assistant says so. In Phase 4 an LLM gets a chance before the refusal, but
the refusal stays as the floor. This is the anti-hallucination design: the
system is allowed to be limited, it is not allowed to be confidently wrong.

**Control commands bypass everything.** `stop`, `cancel`, `never mind` are
matched before any routing, dispatch or permission check, and they cannot be
routed to a model. A stop command that has to wait for inference is not a stop
command.

**`DRY_RUN` on by default.** The first time you run something that can change
your system by voice, you want to read the log first.

---

## What is deliberately absent

There is no code here that clicks a Pay button, types a UPI PIN, an OTP or a
CVV. Not disabled — absent. Phase 5 keeps it that way: the agent researches and
prepares, then hands the last step to you.

---

## Next

Phase 2 is the Tauri overlay (`ambient/state.py` already publishes to
subscribers, which is the hook it needs). Phase 3b is the extractive Q&A
engine — SearxNG plus trafilatura, also CPU-only. Both are buildable on the
laptop you have now. Buy the GPU machine last, when Phase 4 actually needs it.
