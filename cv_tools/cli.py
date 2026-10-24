import json
import os
import platform
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import transformers

from cv_tools import ROOT
from cv_tools.media import IMAGE_SUFFIXES


def parse(parser, tool):
    parser.add_argument("--input", type=Path, required=True, help="Image or video path")
    parser.add_argument("--output", type=Path, help="Fresh run directory; default: runs/<tool>/<timestamp>")
    parser.add_argument("--frames", type=int, help="Frame limit; default: all, or 30 when benchmarking")
    parser.add_argument("--benchmark", action="store_true", help="Two measured runs instead of one")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup passes on the first frame (default: 5)")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                        help="auto selects MPS on macOS and CUDA on Linux; CPU must be requested explicitly")
    args = parser.parse_args()
    args.tool = tool
    args.input = args.input.expanduser().resolve()
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    args.is_image = args.input.suffix.lower() in IMAGE_SUFFIXES
    args.frames = args.frames if args.frames is not None else (30 if args.benchmark else 0)
    if args.frames < 0 or args.warmup < 1:
        parser.error("--frames must be nonnegative and --warmup positive")
    args.out = (args.output or ROOT / "runs" / tool / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).expanduser().resolve()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output is not empty: {args.out}; choose a fresh run directory")
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
    return args


def measure(args, process):
    runs = []
    for index in range(2 if args.benchmark else 1):
        runs.append(process(args.out / f"run-{index + 1}"))
        print(f"run {index + 1}: {runs[-1]['frames']} frames, {runs[-1]['fps']:.2f} FPS", flush=True)
    return runs


def write_summary(args, summary, runs, fps):
    mean_ms = statistics.mean(run["ms_per_frame"] for run in runs)
    summary = {"input": str(args.input), "platform": platform.system(), "device_type": args.device,
               "device": torch.cuda.get_device_name() if args.device == "cuda" else args.device,
               "torch": torch.__version__, "transformers": transformers.__version__,
               "python": platform.python_version(), **summary,
               "runs": runs, "mean_ms_per_frame": mean_ms, "throughput_fps": 1000 / mean_ms}
    if not args.is_image:
        summary |= {"source_fps": fps, "real_time": summary["throughput_fps"] >= fps}
    if args.device == "cuda":
        summary |= {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "process_vram_mib": None}
        # [Agent] TensorRT and cuDNN allocate outside PyTorch's allocator; only the driver sees the process total.
        try:
            rows = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
            summary["process_vram_mib"] = next(
                (int(mib) for pid, mib in (row.split(",") for row in rows.splitlines()) if int(pid) == os.getpid()), None)
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    vram = f", VRAM {summary['process_vram_mib']} MiB" if summary.get("process_vram_mib") else ""
    print(f"{args.tool}: {summary['throughput_fps']:.2f} FPS{vram} -> {args.out}")
