#!/usr/bin/env bash
# One-command demo. Sets up the venv on first run, then reconciles a batch.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "→ creating venv"
  (command -v uv >/dev/null && uv venv .venv) || python3 -m venv .venv
  (command -v uv >/dev/null && uv pip install -q -r requirements.txt) \
    || .venv/bin/pip install -q -r requirements.txt
fi

case "${1:-run}" in
  run)        .venv/bin/python -m khata.cli run "${@:2}" ;;
  offline)    .venv/bin/python -m khata.cli run --no-llm "${@:2}" ;;
  benchmark)  .venv/bin/python -m khata.cli benchmark "${@:2}" ;;
  forecast)   .venv/bin/python -m khata.cli forecast "${@:2}" ;;
  dashboard)  .venv/bin/python -m khata.cli serve "${@:2}" ;;
  test)       .venv/bin/python -m pytest tests/ -q "${@:2}" ;;
  *)          .venv/bin/python -m khata.cli "$@" ;;
esac
