#!/usr/bin/env bash
#
# One-command install for the ambient assistant (spec Phase 9).
#
# Idempotent: safe to re-run to update an existing install. It will reuse the
# venv and fast-forward the checkout rather than starting over.
#
#   curl -fsSL <raw-url>/install.sh | bash
#   ./install.sh --service        # also install + enable the user service
#   ./install.sh --no-torch       # skip Whisper/torch (text + tools only)
#   ./install.sh --no-apt         # skip system packages (no sudo available)
#   ./install.sh --dir ~/apps/aa  # install somewhere other than the default
#
set -euo pipefail

REPO="https://github.com/cursed-goblin/ambient-agent.git"
TARGET="${AMBIENT_HOME:-$HOME/ambient-agent}"
DO_APT=1
DO_TORCH=1
DO_SERVICE=0

say()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\n\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --no-apt)     DO_APT=0 ;;
    --no-torch)   DO_TORCH=0 ;;
    --service)    DO_SERVICE=1 ;;
    --dir)        shift; TARGET="${1:?--dir needs a path}" ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# 1. system packages
# ---------------------------------------------------------------------------
# portaudio + pulseaudio-utils are the mic and the volume control.
# wmctrl / brightnessctl / playerctl are what the window, brightness and media
# tools shell out to. Without them those tools degrade to a clear error rather
# than crashing, so they are recommended but not fatal.
APT_PKGS="git python3 python3-venv python3-pip libportaudio2 pulseaudio-utils \
wmctrl brightnessctl playerctl network-manager"

if [ "$DO_APT" = 1 ]; then
  if command -v apt-get >/dev/null 2>&1; then
    say "Installing system packages"
    note "$APT_PKGS"
    sudo apt-get update -qq
    # shellcheck disable=SC2086
    sudo apt-get install -y -qq $APT_PKGS
  else
    warn "No apt-get here. Install these yourself, then re-run with --no-apt:"
    note "$APT_PKGS"
  fi
else
  say "Skipping system packages (--no-apt)"
fi

command -v python3 >/dev/null 2>&1 || die "python3 not found"

# ---------------------------------------------------------------------------
# 2. the code
# ---------------------------------------------------------------------------
if [ -f "$(dirname "$0")/ambient/main.py" ] && [ "$DO_SERVICE" != 2 ]; then
  # Running from inside a checkout already -- install in place.
  TARGET="$(cd "$(dirname "$0")" && pwd)"
  say "Installing in place: $TARGET"
elif [ -d "$TARGET/.git" ]; then
  say "Updating existing checkout: $TARGET"
  git -C "$TARGET" pull --ff-only || warn "Could not fast-forward; leaving as is."
else
  say "Cloning into $TARGET"
  command -v git >/dev/null 2>&1 || die "git not found"
  git clone --depth 1 "$REPO" "$TARGET"
fi

cd "$TARGET"

# ---------------------------------------------------------------------------
# 3. python environment
# ---------------------------------------------------------------------------
if [ -d .venv ]; then
  say "Reusing existing .venv"
else
  say "Creating .venv"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel

if [ "$DO_TORCH" = 1 ]; then
  # CPU-only torch, installed FIRST so the requirements resolver does not pull
  # the default CUDA build. That default is ~2.5GB and useless on a laptop --
  # it is the most common way this install fails.
  say "Installing CPU-only torch (large download, be patient)"
  python -m pip install --quiet \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio \
    || warn "torch failed. Re-run with --no-torch to skip speech recognition."
else
  say "Skipping torch (--no-torch): no local speech recognition"
  note "The assistant still works via --text and the web UI."
fi

say "Installing requirements"
python -m pip install --quiet -r requirements.txt

chmod +x run.sh 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. sanity check
# ---------------------------------------------------------------------------
say "Checking the install"
python - <<'PY'
import sys
try:
    from ambient import gate, tools
except Exception as exc:
    print(f"    FAILED to import: {exc}")
    sys.exit(1)
g = gate.Gate()
print(f"    tools registered: {len(tools.SCHEMAS)}")
print(f"    risk tiers:       {sorted(set(tools.RISK.values()))}")
print(f"    dry run:          {g.dry_run}")
print(f"    confirm all:      {g.confirm_everything}")
PY

# ---------------------------------------------------------------------------
# 5. optional user service
# ---------------------------------------------------------------------------
if [ "$DO_SERVICE" = 1 ]; then
  say "Installing the user service"
  UNIT_DIR="$HOME/.config/systemd/user"
  mkdir -p "$UNIT_DIR"
  sed "s|@HOME@|$TARGET|g" systemd/ambient-agent.service \
    > "$UNIT_DIR/ambient-agent.service"
  # The assistant controls the desktop, so it needs the graphical session's
  # environment. Without this the unit starts but cannot reach your display.
  systemctl --user import-environment DISPLAY XAUTHORITY XDG_RUNTIME_DIR || true
  systemctl --user daemon-reload
  systemctl --user enable ambient-agent.service
  note "Start it:   systemctl --user start ambient-agent"
  note "Watch it:   journalctl --user -u ambient-agent -f"
fi

# ---------------------------------------------------------------------------
say "Done."
cat <<EOF

    cd $TARGET
    ./run.sh --setup        # paste your Groq API key (free tier is fine)
    ./run.sh --check-ai     # confirm the model answers
    ./run.sh --ui           # open http://127.0.0.1:8765

  Safety mode (AMBIENT_DRY_RUN=1) is ON by default: actions are logged,
  not performed. Turn it off in Settings once you trust it.

    ./run.sh --audit        # see everything it has decided to do

EOF
