"""Depth Anything V2: relative depth, near = bright"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

from cv_tools import MODEL_DIR
from cv_tools.cli import measure, parse, write_summary
from cv_tools.media import get_fps, read_frames, write_run

MODELS = {"small": "depth-anything-v2-small", "base": "depth-anything-v2-base", "large": "depth-anything-v2-large"}
COLORMAPS = {"inferno": cv2.COLORMAP_INFERNO, "turbo": cv2.COLORMAP_TURBO, "magma": cv2.COLORMAP_MAGMA,
             "viridis": cv2.COLORMAP_VIRIDIS, "gray": None}
# [Agent] ImageNet statistics; every Depth Anything V2 checkpoint's preprocessor_config.json uses them.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def predict(model, image, args):
    h, w = image.shape[:2]
    # [Agent] Resize and normalize on the GPU: the HF processor's PIL bicubic resize takes ~6 ms per 720p frame
    # [Agent] on CPU, half the small model's forward. Output matches it (Pearson 0.99998). BGR flips to RGB here.
    pixels = torch.from_numpy(image).to(args.device).permute(2, 0, 1)[None].flip(1).float() / 255
    pixels = F.interpolate(pixels, args.size, mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(MEAN, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=args.device).view(1, 3, 1, 1)
    depth = model(pixel_values=((pixels - mean) / std).to(model.dtype)).predicted_depth
    depth = F.interpolate(depth[:, None].float(), (h, w), mode="bilinear", align_corners=False)[0, 0]
    return depth.cpu().numpy()


def process(model, args, destination, fps):
    colormap = COLORMAPS[args.colormap]

    def render(item, index):
        frame, depth = item
        low, high = float(depth.min()), float(depth.max())
        # [Agent] The model predicts relative inverse depth: larger is nearer, with no metric scale. Stretch each
        # [Agent] frame to its own range so near is white, and record that range so the 8/16-bit files map back.
        unit = (depth - low) / max(high - low, 1e-6)
        gray = np.rint(unit * 255).astype(np.uint8)
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if colormap is None else cv2.applyColorMap(gray, colormap)
        record = {"frame": index, "timestamp_seconds": frame.timestamp, "output_timestamp_seconds": index / fps,
                  "min": low, "max": high}
        if args.is_image:
            return record, {"depth.png": np.rint(unit * 65535).astype(np.uint16), "color.jpg": color,
                            "depth.npy": depth}
        return record, {"depth.mp4": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "color.mp4": color}

    items = ((frame, predict(model, frame.image, args)) for frame in read_frames(args.input, args.frames, fps))
    return write_run(destination, items, render, fps, "depth.jsonl")


@torch.inference_mode()
def run(args):
    if args.resolution < 14 or args.resolution % 14:
        raise SystemExit("--resolution must be a positive multiple of 14")
    args.model = MODEL_DIR / MODELS[args.model] if args.model in MODELS else Path(args.model).expanduser()
    if not (args.model / "config.json").is_file():
        raise SystemExit(f"No checkpoint at {args.model}; see README › Installation")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    first = next(read_frames(args.input, 1)).image
    h, w = first.shape[:2]
    scale = args.resolution / min(h, w)
    # [Agent] Short side to --resolution, long side keeps the aspect ratio; both snap to the 14-pixel ViT patch.
    args.size = (max(14, round(h * scale / 14) * 14), max(14, round(w * scale / 14) * 14))
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    print(f"depth-anything: {args.device}, {args.model.name}, model input {args.size[0]}x{args.size[1]}", flush=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    # [Agent] Plain PyTorch on purpose: an FP16 TensorRT engine with FP32 norm guards measured 155 ms vs 133 ms
    # [Agent] for large on the 3090, and the backbone already runs SDPA attention.
    model = AutoModelForDepthEstimation.from_pretrained(args.model, local_files_only=True, dtype=dtype)
    model = model.to(args.device).eval()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(args.warmup):
        predict(model, first, args)
    warmup_seconds = time.perf_counter() - started
    fps = get_fps(args.input)
    runs = measure(args, lambda destination: process(model, args, destination, fps))
    summary = {"model": str(args.model), "dtype": str(dtype), "size_hw": args.size, "colormap": args.colormap,
               "load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
               "warmup_passes": args.warmup}
    write_summary(args, summary, runs, fps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="large", help="small | base | large, or a local HF checkpoint path (default: large)")
    p.add_argument("--resolution", type=int, default=518,
                   help="Model input short side, a multiple of 14 (default: 518, the training size)")
    p.add_argument("--colormap", choices=list(COLORMAPS), default="inferno")
    run(parse(p, "depth-anything"))


if __name__ == "__main__":
    main()
