"""Colour match: take a render's colours to its source clip's.

Default, the grade: per-channel tone curves fitted on the background only (outside motion_guide's person mask, grown),
printed as an ffmpeg `curves` filter for the whole render. A swapped person is not supposed to match the source.

--inside KEY: inside one masks.npz mask (the face), every frame's skin moves to the source's by the mean Lab a*/b*
difference. Brightness and everything outside stay the render's. Writes --out.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from cv_tools.media import get_fps

CURVE_POINTS = 17
TONE = {"smooth": 0.5, "erode": 7, "feather": 41}  # median window in s; px the mask shrinks (sample width); px of edge blur (render)
EXACT = "accurate_rnd+full_chroma_int"


def get_size(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                                   "-of", "csv=p=0", str(path)], text=True)
    return tuple(map(int, out.strip().split(",")[:2]))


def read_frames(path, size=None, fps=None):
    """BGR uint8 frames, scaled to `size` (w, h) and resampled to `fps` when given."""
    w, h = size or get_size(path)
    # [Agent] Exact rounding: ffmpeg's default YUV->RGB rounding darkened every pixel by about 2 levels, so a toned render
    # [Agent] came out shifted everywhere, hair and background included.
    scale = (f"fps={fps}," if fps else "") + f"scale={w}:{h}:flags={EXACT}"
    ffmpeg = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf", scale, "-f", "rawvideo",
                               "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    # [Agent] Callers stop early (a source longer than its render): kill the decoder, or it dies on the closed pipe and says so.
    try:
        while chunk := ffmpeg.stdout.read(w * h * 3):
            yield np.frombuffer(chunk, np.uint8).reshape(h, w, 3)
    finally:
        ffmpeg.kill()
        ffmpeg.wait()


def background(masks, index, size, grow):
    mask = cv2.resize(masks[min(index, len(masks) - 1)].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    # [Agent] Grow the person mask well past her outline: the new person's hair and arms land a little differently from
    # [Agent] the source's, and any of it left in the sample drags the curves toward the wrong colour.
    return cv2.dilate(mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) == 0


def fit(source, render):
    q = np.linspace(0, 1, CURVE_POINTS)
    xs = np.linspace(0, 255, CURVE_POINTS)
    # [Agent] Quantile matching: the render's p-th brightness level goes to the source's p-th level. Reading that map at
    # [Agent] even input steps gives the curve; black stays black and the running max keeps it from ever going down.
    ys = np.interp(xs, np.quantile(render, q), np.quantile(source, q), left=0, right=np.quantile(source, 1))
    return xs / 255, np.maximum.accumulate(ys) / 255


def grade(args):
    w, h = get_size(args.source)
    size = (args.width, round(args.width * h / w))
    masks = np.load(args.masks)["masks"] if args.masks else None
    source_fps, rate = get_fps(args.source), 1 / args.step
    source_px, render_px, samples = [], [], 0
    for n, (src, ren) in enumerate(zip(read_frames(args.source, size, rate), read_frames(args.render, size, rate))):
        t = n * args.step
        if t >= args.end:
            break
        if t < args.start:
            continue
        # [Agent] Letterbox bars and clipped highlights are flat in the source and say nothing about its grade.
        keep = (src.max(axis=2) > 12) & (src.max(axis=2) < 250)
        if masks is not None:
            keep &= background(masks, round(t * source_fps), size, args.grow)
        source_px.append(src[keep])
        render_px.append(ren[keep])
        samples += 1
    if samples < len(np.arange(args.start, args.end, args.step)):
        raise SystemExit(f"color_match: a clip ends before {args.end} s; lower --end")
    source_px, render_px = np.concatenate(source_px), np.concatenate(render_px)

    channels, shifts = [], []
    for name, c in (("r", 2), ("g", 1), ("b", 0)):  # BGR frames, r g b curves
        xs, ys = fit(source_px[:, c], render_px[:, c])
        # [Agent] How far the full fit strays from no change, in 0-255 levels, measured before --strength so a clip always
        # [Agent] reads the same number: it says whether matching is worth it.
        shifts.append(f"{name} {np.abs(ys - xs).max() * 255:.0f}")
        ys = xs + args.strength * (ys - xs)
        channels.append(f"{name}='" + " ".join(f"{x:.4f}/{y:.4f}" for x, y in zip(xs, ys)) + "'")
    print(f"color_match: {samples} samples, {len(source_px)} px, largest shift {', '.join(shifts)} levels", file=sys.stderr)
    print("curves=" + ":".join(channels))


def to_lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def weigh_skin(lab):
    # [Agent] Skin is colourful; grey or ash hair, brows, lashes and eye whites are not. Chroma under 10 weighs 0, over 20 weighs 1.
    return np.clip((np.hypot(lab[..., 1], lab[..., 2]) - 10) / 10, 0, 1)


def get_mask(masks, i, ratio, size):
    """The source's mask under render frame i (ratio = source fps / render fps), at render size."""
    return cv2.resize(masks[min(round(i * ratio), len(masks) - 1)].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)


def average_skin(frame, core):
    """Mean (a*, b*) of the skin inside `core`; NaN under 200 px of skin."""
    lab = to_lab(frame)
    weight = weigh_skin(lab) * core
    return (lab[..., 1:] * weight[..., None]).sum((0, 1)) / weight.sum() if weight.sum() > 200 else np.full(2, np.nan)


def measure(source, render, masks, ratio, size):
    """Per render frame: the source's skin minus the render's, inside the shrunk mask."""
    kernel = np.ones((TONE["erode"],) * 2, np.uint8)
    shifts = []
    for i, ren in enumerate(render):
        src = next(source, None)
        core = cv2.erode(get_mask(masks, i, ratio, size), kernel) > 0
        shifts.append(np.full(2, np.nan) if src is None else average_skin(src, core) - average_skin(ren, core))
    return np.array(shifts)


def smooth(shifts, fps):
    """Frames without a measure take the nearest one; a moving median over TONE["smooth"] s keeps a light change and drops flicker."""
    found = ~np.isnan(shifts[:, 0])
    index = np.arange(len(shifts))
    filled = np.stack([np.interp(index, index[found], shifts[found, c]) for c in range(2)], 1)
    half = round(TONE["smooth"] * fps) // 2
    return np.array([np.median(filled[max(0, i - half):i + half + 1], 0) for i in index])


def shift(frame, mask, ab):
    """The frame with the skin under the feathered mask moved by ab (a*, b*); every other pixel keeps its exact bytes."""
    lab = to_lab(frame)
    weight = cv2.GaussianBlur(mask.astype(np.float32), (TONE["feather"],) * 2, 0) * weigh_skin(lab)
    lab[..., 1:] += weight[..., None] * ab
    moved = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)
    return np.where(weight[..., None] > 0, moved, frame)


def tone(args):
    masks = np.load(args.masks)
    if args.inside not in masks.files:
        raise SystemExit(f"{args.masks} has {', '.join(masks.files)}, not {args.inside}; motion_guide --color saves face and hair")
    masks = masks[args.inside]
    size, fps = get_size(args.render), get_fps(args.render)
    sample = (args.width, round(args.width * size[1] / size[0]))
    ratio = get_fps(args.source) / fps
    # [Agent] Both clips are measured at sample width (a mean over a face needs no more pixels; full size took 3x as long),
    # [Agent] and the source at the render's frame rate, so frame i of each shows the same moment.
    shifts = measure(read_frames(args.source, sample, fps), read_frames(args.render, sample), masks, ratio, sample)
    found = int((~np.isnan(shifts[:, 0])).sum())
    if not found:
        raise SystemExit(f"color_match: too little skin inside {args.inside} on every frame")
    shifts = args.strength * smooth(shifts, fps)
    print(f"color_match: {args.inside} measured on {found}/{len(shifts)} frames, shift red {shifts[:, 0].min():+.1f}.."
          f"{shifts[:, 0].max():+.1f}, yellow {shifts[:, 1].min():+.1f}..{shifts[:, 1].max():+.1f} (Lab a*, b*)", file=sys.stderr)

    w, h = size
    encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
                                "-i", "-", "-i", str(args.render), "-map", "0:v", "-map", "1:a?",
                                "-vf", f"scale=flags={EXACT}:out_color_matrix=bt709:out_range=tv", "-colorspace", "bt709",
                                "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                                "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "copy", str(args.out)],
                               stdin=subprocess.PIPE)
    for i, frame in enumerate(read_frames(args.render)):
        encoder.stdin.write(shift(frame, get_mask(masks, i, ratio, size), shifts[i]).tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError(f"ffmpeg failed writing {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="The clip the render was made from")
    p.add_argument("--render", type=Path, required=True, help="The render to match, same timing as --source")
    p.add_argument("--masks", type=Path, help="motion_guide's masks.npz for --source; grade default: every pixel counts")
    p.add_argument("--inside", help="Tone inside this masks.npz mask (face, hair: motion_guide --color) and write --out")
    p.add_argument("--out", type=Path, help="With --inside: the toned render (.mp4, the render's sound copied)")
    p.add_argument("--start", type=float, default=0.0, help="Grade: first second sampled (default 0)")
    p.add_argument("--end", type=float, help="Grade, required: last second sampled; leave out dissolves and fades")
    p.add_argument("--step", type=float, default=0.25, help="Grade: seconds between samples (default 0.25)")
    p.add_argument("--width", type=int, default=320, help="Sample width in pixels (default 320)")
    p.add_argument("--grow", type=int, default=18, help="Grade: pixels the person mask grows at sample width (default 18)")
    p.add_argument("--strength", type=float, default=1.0, help="1 = full match, 0.5 = halfway back to no change (default 1)")
    args = p.parse_args()
    if args.inside and not (args.masks and args.out):
        p.error("--inside needs --masks and --out")
    if args.inside:
        tone(args)
    elif args.end is None:
        p.error("the grade needs --end")
    else:
        grade(args)


if __name__ == "__main__":
    main()
