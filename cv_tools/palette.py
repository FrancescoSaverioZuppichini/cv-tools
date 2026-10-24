"""Palette: dominant colours plus highlight and shadow tint, for lighting references"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from cv_tools.media import get_fps, read_frames

PIXELS = 65536


def sample(path, every):
    # [Agent] Downscale each frame to ~64k pixels: colour statistics barely change, and k-means stays ~50 ms.
    images = []
    for frame in read_frames(path, 0):
        if frame.index % every == 0:
            h, w = frame.image.shape[:2]
            scale = min(1.0, (PIXELS / (h * w)) ** 0.5)
            images.append(cv2.resize(frame.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    if not images:
        raise SystemExit(f"No readable frames in {path}")
    # [Agent] Lab distances follow perceived colour difference, so clusters split the way a person sees them.
    return np.concatenate([cv2.cvtColor(i.astype(np.float32) / 255, cv2.COLOR_BGR2Lab).reshape(-1, 3) for i in images])


def get_color(lab, share=None):
    bgr = cv2.cvtColor(np.float32(lab).reshape(1, 1, 3), cv2.COLOR_Lab2BGR)[0, 0]
    r, g, b = (int(round(v)) for v in np.clip(bgr[::-1] * 255, 0, 255))
    color = {"hex": f"#{r:02X}{g:02X}{b:02X}", "rgb": [r, g, b], "lightness": round(float(lab[0]), 1)}
    return color if share is None else {**color, "share": round(float(share), 4)}


def extract(lab, colors):
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centers = cv2.kmeans(lab, colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=colors) / len(labels)
    order = np.argsort(-shares)
    lightness = lab[:, 0]
    return {"palette": [get_color(centers[i], shares[i]) for i in order],
            "highlights": get_color(lab[lightness >= np.percentile(lightness, 90)].mean(axis=0)),
            "shadows": get_color(lab[lightness <= np.percentile(lightness, 10)].mean(axis=0)),
            "mean_lightness": round(float(lightness.mean()), 1)}


def show(result, fmt):
    if fmt == "json":
        return json.dumps(result, indent=2)
    if fmt == "hex":
        return "\n".join(c["hex"] for c in result["palette"])
    if fmt == "rgb":
        return "\n".join(", ".join(map(str, c["rgb"])) for c in result["palette"])
    swatch = sys.stdout.isatty()

    def line(label, c):
        block = f"\033[48;2;{c['rgb'][0]};{c['rgb'][1]};{c['rgb'][2]}m    \033[0m " if swatch else ""
        share = f"{c['share'] * 100:5.1f}%" if "share" in c else ""
        return f"{block}{label:<10} {c['hex']}  rgb({', '.join(f'{v:3d}' for v in c['rgb'])})  L {c['lightness']:5.1f}  {share}".rstrip()

    rows = [line("", c) for c in result["palette"]]
    rows += ["", line("highlight", result["highlights"]), line("shadow", result["shadows"]),
             f"mean lightness {result['mean_lightness']} / 100"]
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Image or video path")
    p.add_argument("--colors", type=int, default=6, help="Palette size (default: 6)")
    p.add_argument("--format", choices=["text", "hex", "rgb", "json"], default="text",
                   help="text table with swatches (default), one hex or r,g,b per line, or JSON with everything")
    p.add_argument("--sample-fps", type=float, default=1.0, help="Video frames sampled per second (default: 1)")
    args = p.parse_args()
    if not args.input.is_file():
        p.error(f"Input does not exist: {args.input}")
    if not 1 <= args.colors <= 32 or args.sample_fps <= 0:
        p.error("--colors must be 1-32 and --sample-fps positive")
    every = max(1, round(get_fps(args.input) / args.sample_fps))
    print(show(extract(sample(args.input, every), args.colors), args.format))


if __name__ == "__main__":
    main()
