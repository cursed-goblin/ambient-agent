#!/usr/bin/env bash
# Load PipeWire's WebRTC echo canceller.
#
# This is the single most important step in the whole project. Without AEC the
# assistant hears its own voice through the laptop speakers, and barge-in
# becomes impossible: it interrupts itself on every reply.
#
# Verify by running tools/check_audio.py afterwards.

set -euo pipefail

CONF_DIR="$HOME/.config/pipewire/pipewire.conf.d"
CONF_FILE="$CONF_DIR/99-echo-cancel.conf"

echo "==> Checking PipeWire"
if ! command -v pw-cli >/dev/null 2>&1; then
  echo "PipeWire not found. On Ubuntu 24.04 it is the default; install with:"
  echo "  sudo apt install pipewire pipewire-pulse wireplumber pulseaudio-utils"
  exit 1
fi
pw-cli info 0 >/dev/null 2>&1 || { echo "PipeWire is not running."; exit 1; }
echo "    ok"

echo "==> Writing $CONF_FILE"
mkdir -p "$CONF_DIR"
cat > "$CONF_FILE" <<'EOF'
context.modules = [
    {   name = libpipewire-module-echo-cancel
        args = {
            # Leave source/sink names unset so WirePlumber picks the defaults.
            library.name  = aec/libspa-aec-webrtc
            capture.props = {
                node.name   = "echo_cancel.capture"
            }
            source.props = {
                node.name        = "echo_cancel.source"
                node.description = "Echo-Cancelled Mic (ambient-agent)"
            }
            sink.props = {
                node.name        = "echo_cancel.sink"
                node.description = "Echo-Cancelled Output (ambient-agent)"
            }
            playback.props = {
                node.name   = "echo_cancel.playback"
            }
            aec.args = {
                webrtc.gain_control          = true
                webrtc.extended_filter       = true
                webrtc.high_pass_filter      = true
                webrtc.noise_suppression     = true
                webrtc.voice_detection       = true
                webrtc.experimental_agc      = true
            }
        }
    }
]
EOF
echo "    written"

echo "==> Restarting PipeWire"
systemctl --user restart pipewire pipewire-pulse wireplumber 2>/dev/null || true
sleep 3

echo "==> Available sources"
if command -v pactl >/dev/null 2>&1; then
  pactl list short sources || true
fi

if pactl list short sources 2>/dev/null | grep -q echo_cancel; then
  echo
  echo "AEC source is live."
  echo
  echo "Make it the default input:"
  echo "  pactl set-default-source echo_cancel.source"
  echo "  pactl set-default-sink   echo_cancel.sink"
  echo
  echo "Or pin it in the app instead, without touching system defaults:"
  echo "  export AMBIENT_INPUT_DEVICE=echo_cancel"
  echo "  export AMBIENT_OUTPUT_DEVICE=echo_cancel"
  echo
  echo "Then: python3 tools/check_audio.py"
else
  echo
  echo "echo_cancel source did NOT appear. Check:  journalctl --user -u pipewire -n 50"
  exit 1
fi
