#!/bin/sh
# macOS / Linux: start the game backend (dashboard at http://127.0.0.1:8765)
# Uses its own Python environment in ./venv_mac (created automatically the first time).
cd "$(dirname "$0")"
if [ ! -x venv_mac/bin/python ]; then
  echo "First run: creating Python environment (venv_mac) ..."
  python3 -m venv venv_mac || exit 1
fi
if ! venv_mac/bin/python -c "import matplotlib, numpy" 2>/dev/null; then
  echo "Installing matplotlib + numpy ..."
  venv_mac/bin/python -m pip install --upgrade pip -q
  venv_mac/bin/python -m pip install matplotlib numpy || exit 1
fi
if [ "$1" = "--with-rl" ] && ! venv_mac/bin/python -c "import torch" 2>/dev/null; then
  echo "Installing torch for the RL agent (large download) ..."
  venv_mac/bin/python -m pip install torch
fi
exec venv_mac/bin/python backend/server.py
