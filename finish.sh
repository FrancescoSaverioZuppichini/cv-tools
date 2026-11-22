#!/bin/bash
# finish.sh <folder or images> [flags]: phone finish (sharpen x1 -> real iPhone noise D 28 % -> JPEG 95, same size).
# Flags pass through to python -m cv_tools.phone_finish (--help). Temp files go under this checkout (runs/tmp), never the root disk.
args=(); while [ $# -gt 0 ] && [[ $1 != --* ]]; do args+=("$(realpath "$1")"); shift; done
cd "$(dirname "$(realpath "$0")")"
export CV_TOOLS_DATA="${CV_TOOLS_DATA:-$PWD}" TMPDIR="${TMPDIR:-$PWD/runs/tmp}"
mkdir -p "$TMPDIR"
exec .venv/bin/python -W ignore -m cv_tools.phone_finish --input "${args[@]}" "$@"
