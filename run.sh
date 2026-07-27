#!/usr/bin/env bash
# Launch the Phase 1 ambient loop.
#
#   ./run.sh              full voice loop
#   ./run.sh --text       type commands, no mic/speaker needed
#   ./run.sh --devices    list audio devices
#   ./run.sh --say "hi"   TTS smoke test

set -euo pipefail
cd "$(dirname "$0")"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Pin the echo-cancelled devices without changing system defaults.
export AMBIENT_INPUT_DEVICE="${AMBIENT_INPUT_DEVICE:-echo_cancel}"
export AMBIENT_OUTPUT_DEVICE="${AMBIENT_OUTPUT_DEVICE:-echo_cancel}"

# Safety default: DRY_RUN=1 means no OS command actually executes; every
# action is logged instead. Set AMBIENT_DRY_RUN=0 when you trust it.
export AMBIENT_DRY_RUN="${AMBIENT_DRY_RUN:-1}"

if [ "$AMBIENT_DRY_RUN" = "1" ]; then
  echo "[safety] DRY_RUN is ON -- actions are logged, not executed."
  echo "[safety] Disable with: AMBIENT_DRY_RUN=0 ./run.sh"
  echo
fi

exec python3 -m ambient.main "$@"
