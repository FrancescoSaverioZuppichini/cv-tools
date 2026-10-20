import argparse
import platform
import re
from datetime import datetime
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ROOT = Path(__file__).resolve().parents[1]


def resolution(value):
    value = str(value).strip().lower().replace("×", "x")
    aliases = {"full": (1024, 768), "native": (1024, 768), "1": (1024, 768),
               "half": (512, 384), "1/2": (512, 384), "quarter": (256, 192), "1/4": (256, 192)}
    if value in aliases:
        return aliases[value]
    if re.fullmatch(r"\d+", value):
        height = int(value)
        size = (height, max(32, round(height * 0.75 / 32) * 32))
    elif re.fullmatch(r"\d+x\d+", value):
        size = tuple(map(int, value.split("x")))
    else:
        raise argparse.ArgumentTypeError("resolution expects full, half, 1/2, quarter, 1/4, a height like 512, or HxW like 512x384")
    if min(size) < 32 or any(n % 32 for n in size):
        raise argparse.ArgumentTypeError("resolution dimensions must be at least 32 and divisible by 32")
    return size


def add_input_arguments(parser):
    parser.add_argument("--input", type=Path, required=True, help="Image or video path")
    parser.add_argument("--output", type=Path, help="Run directory; default: runs/<model>/<timestamp>")
    parser.add_argument("--frames", type=int, help="Frame limit; default: all, or 30 when benchmarking")
    parser.add_argument("--benchmark", action="store_true", help="Five warmup batches, two measured runs")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--resolution", type=resolution, default=(1024, 768),
                        help="full | half | 1/2 | quarter | 1/4 | HEIGHT | HxW (default: full)")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                        help="auto selects MPS on macOS and CUDA on Linux; CPU must be requested explicitly")


def resolve_device(args, parser):
    import torch

    system = platform.system()
    args.device = ("mps" if system == "Darwin" else "cuda") if args.device == "auto" else args.device
    if args.device == "mps" and system != "Darwin":
        parser.error(f"MPS requires macOS; this machine is {system}. Use --device cuda or cpu.")
    if args.device == "cuda" and system == "Darwin":
        parser.error("CUDA is unavailable on macOS. Use --device mps or cpu.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple MPS is unavailable in this PyTorch build or on this Mac. Use --device cpu explicitly.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable: check the NVIDIA GPU, driver, and CUDA-enabled PyTorch, or use --device cpu.")


def resolve_input_arguments(args, model, parser):
    args.input = args.input.expanduser().resolve()
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    args.frames = args.frames if args.frames is not None else (30 if args.benchmark else 0)
    args.size = args.resolution
    args.out = (args.output or ROOT / "runs" / model / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).expanduser().resolve()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output is not empty: {args.out}; choose a fresh run directory")
