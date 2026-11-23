"""Colour match: fit per-channel tone curves that take a render's colours to its source's, on background pixels only.

Prints an ffmpeg `curves` filter for the render. The person mask from motion_guide (masks.npz) keeps the person out
of the fit, since a swapped person (new hair, new skin) is not supposed to match the source.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from cv_tools.media import get_fps, read_frames

POINTS = 17


def sample(path, times, size):
    # [Agent] Decode once from the start and keep the frames nearest the sample times; seeking per sample restarts the
    # [Agent] decoder at a keyframe each time, which was 80 % of the old script's time.
    fps = get_fps(path)
    wanted = {round(t * fps): i for i, t in enumerate(times)}
    frames = [None] * len(times)
    for frame in read_frames(path, max(wanted) + 1, fps):
        if frame.index in wanted:
            frames[wanted[frame.index]] = cv2.resize(frame.image, size, interpolation=cv2.INTER_AREA)
    missing = [t for t, f in zip(times, frames) if f is None]
    if missing:
        raise ValueError(f"{path} ends before {missing[0]:.2f} s; lower --end")
    return frames


def background(masks, index, size, grow):
    mask = cv2.resize(masks[min(index, len(masks) - 1)].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    # [Agent] Grow the person mask well past her outline: the new person's hair and arms land a little differently from
    # [Agent] the source's, and any of it left in the sample drags the curves toward the wrong colour.
    return cv2.dilate(mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) == 0


def fit(source, render):
    q = np.linspace(0, 1, POINTS)
    xs = np.linspace(0, 255, POINTS)
    # [Agent] Quantile matching: the render's p-th brightness level goes to the source's p-th level. Reading that map at
    # [Agent] even input steps gives the curve; black stays black and the running max keeps it from ever going down.
    ys = np.interp(xs, np.quantile(render, q), np.quantile(source, q), left=0, right=np.quantile(source, 1))
    return xs / 255, np.maximum.accumulate(ys) / 255


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="The clip the render was made from")
    p.add_argument("--render", type=Path, required=True, help="The render to grade, same timing as --source")
    p.add_argument("--masks", type=Path, help="motion_guide's masks.npz for --source; default: every pixel counts")
    p.add_argument("--start", type=float, default=0.0, help="First second sampled (default 0)")
    p.add_argument("--end", type=float, required=True, help="Last second sampled; leave out dissolves and fades")
    p.add_argument("--step", type=float, default=0.25, help="Seconds between samples (default 0.25)")
    p.add_argument("--width", type=int, default=320, help="Sample width in pixels (default 320)")
    p.add_argument("--grow", type=int, default=18, help="Pixels the person mask grows at sample width (default 18)")
    p.add_argument("--strength", type=float, default=1.0, help="1 = full match, 0.5 = every curve point halfway back to no change (default 1)")
    args = p.parse_args()

    height, width = cv2.VideoCapture(str(args.source)).read()[1].shape[:2]
    size = (args.width, round(args.width * height / width))
    times = list(np.arange(args.start, args.end, args.step))
    masks = np.load(args.masks)["masks"] if args.masks else None
    source_fps = get_fps(args.source)

    source_px, render_px = [], []
    for t, src, ren in zip(times, sample(args.source, times, size), sample(args.render, times, size)):
        # [Agent] Letterbox bars and clipped highlights are flat in the source and say nothing about its grade.
        keep = (src.max(axis=2) > 12) & (src.max(axis=2) < 250)
        if masks is not None:
            keep &= background(masks, round(t * source_fps), size, args.grow)
        source_px.append(src[keep])
        render_px.append(ren[keep])
    source_px, render_px = np.concatenate(source_px), np.concatenate(render_px)

    # [Agent] Frames are BGR; ffmpeg's curves filter wants r, g, b.
    channels, shifts = [], []
    for name, c in (("r", 2), ("g", 1), ("b", 0)):
        xs, ys = fit(source_px[:, c], render_px[:, c])
        # [Agent] How far the full fit strays from no change, in 0-255 levels: the number that says whether matching is worth it.
        # [Agent] It is measured before --strength, so the same clip always reads the same number.
        shift = np.abs(ys - xs).max() * 255
        ys = xs + args.strength * (ys - xs)
        channels.append(f"{name}='" + " ".join(f"{x:.4f}/{y:.4f}" for x, y in zip(xs, ys)) + "'")
        shifts.append(f"{name} {shift:.0f}")
    print(f"color_match: {len(times)} samples, {len(source_px)} px, largest shift {', '.join(shifts)} levels", file=sys.stderr)
    print("curves=" + ":".join(channels))


if __name__ == "__main__":
    main()
