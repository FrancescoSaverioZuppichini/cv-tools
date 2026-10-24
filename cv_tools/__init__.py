import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# [Agent] Checkpoints and TensorRT engines are large and machine-specific, so CV_TOOLS_DATA can put both on a data drive
# while every command and default stays the same.
DATA = Path(os.environ.get("CV_TOOLS_DATA", ROOT)).expanduser()
MODEL_DIR = DATA / ".models"
ENGINE_DIR = DATA / "engines"
