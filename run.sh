#!/usr/bin/env sh
# macOS and Linux: sets up on first run, then starts the walkthrough and opens the browser.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up, needs internet once..."
  python3 -m venv .venv || exit 1
  if [ "$(uname)" = "Darwin" ]; then
    .venv/bin/python -m pip install torch==2.14.1 || exit 1
  else
    .venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu || exit 1
  fi
  .venv/bin/python -m pip install -r requirements.txt || exit 1
fi
exec .venv/bin/python server.py --open
