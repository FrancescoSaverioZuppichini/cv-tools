"""Canny: edge map of an image or video, white edges on black"""
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from cv_tools import ROOT
from cv_tools.media import IMAGE_SUFFIXES, get_fps, read_frames, write_run


def edges(image, low, high, sigma, blur):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur:
        # [Agent] A light blur first, so skin and fabric texture do not turn into speckle edges.
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    if low is None or high is None:
        # [Agent] Auto thresholds from the median brightness (median ± sigma), the usual auto-Canny rule: one setting
        # [Agent] works on dark night shots and bright studio shots alike, and explicit --low/--high override it.
        median = float(np.median(gray))
        low, high = int(max(0, (1 - sigma) * median)), int(min(255, (1 + sigma) * median))
    return cv2.Canny(gray, low, high), low, high


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Image or video path")
    p.add_argument("--output", type=Path, help="Fresh run directory; default: runs/canny/<timestamp>")
    p.add_argument("--low", type=int, help="Lower hysteresis threshold; default: auto from the median")
    p.add_argument("--high", type=int, help="Upper hysteresis threshold; default: auto from the median")
    p.add_argument("--sigma", type=float, default=0.33, help="Auto-threshold spread around the median (default: 0.33)")
    p.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel before edges, odd; 0 disables (default: 5)")
    p.add_argument("--frames", type=int, default=0, help="Frame limit for videos; default: all")
    args = p.parse_args()
    args.input = args.input.expanduser().resolve()
    if not args.input.is_file():
        p.error(f"Input does not exist: {args.input}")
    if args.blur and args.blur % 2 == 0:
        p.error("--blur must be odd or 0")
    out = (args.output or ROOT / "runs" / "canny" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        p.error(f"Output is not empty: {out}; choose a fresh run directory")
    is_image = args.input.suffix.lower() in IMAGE_SUFFIXES
    fps = get_fps(args.input)

    def render(frame, index):
        edge, low, high = edges(frame.image, args.low, args.high, args.sigma, args.blur)
        record = {"frame": index, "timestamp_seconds": frame.timestamp, "low": low, "high": high,
                  "edge_pixels": int(np.count_nonzero(edge))}
        return record, {("canny.png" if is_image else "canny.mp4"): edge if is_image else cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)}

    stats = write_run(out, read_frames(args.input, args.frames, fps), render, fps, "edges.jsonl")
    print(f"canny: {stats['frames']} frames, {stats['fps']:.1f} FPS -> {out}")


if __name__ == "__main__":
    main()
