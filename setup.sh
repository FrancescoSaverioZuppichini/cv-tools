#!/bin/bash
# One environment for every tool: .venv (or $VENV), plus Meta's SAM 3 and the PNG noise code in third_party/.
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
VENV="${VENV:-$ROOT/.venv}"
"${PYTHON:-python3}" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q uv
UV=("$VENV/bin/uv" pip install --python "$VENV/bin/python")
if [[ "$(uname -s)" == "Darwin" ]]; then
  "${UV[@]}" -e "$ROOT"
else
  "${UV[@]}" torch==2.15.0 "torchvision>=0.29,<0.30" --index-url https://download.pytorch.org/whl/cu132
  "${UV[@]}" -e "$ROOT[tensorrt]"
fi
# [Agent] Both repos are used as code: SAM 3 installs without its pins, PNG has no package metadata and is imported from its folder.
[[ -d "$ROOT/third_party/sam3" ]] || git clone --depth 1 https://github.com/facebookresearch/sam3 "$ROOT/third_party/sam3"
[[ -d "$ROOT/third_party/png" ]] || git clone --depth 1 https://github.com/JK-the-Ko/PNG "$ROOT/third_party/png"
"${UV[@]}" --no-deps -e "$ROOT/third_party/sam3"
