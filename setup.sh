#!/usr/bin/env bash
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
if [[ "$(uname -s)" == "Darwin" ]]; then
  "${PYTHON:-python3}" -m venv "$ROOT/.venv"
else
  "${PYTHON:-python3.12}" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install uv
if [[ "$(uname -s)" == "Darwin" ]]; then
  "$ROOT/.venv/bin/uv" pip install --python "$ROOT/.venv/bin/python" -e "$ROOT"
else
  "$ROOT/.venv/bin/uv" pip install --python "$ROOT/.venv/bin/python" \
    torch==2.11.0 torchvision==0.26.0 --index https://download.pytorch.org/whl/cu128
  "$ROOT/.venv/bin/uv" pip install --python "$ROOT/.venv/bin/python" -e "$ROOT[tensorrt]"
fi
