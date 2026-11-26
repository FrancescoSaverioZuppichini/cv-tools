"""Colour match: take a render's colours to its source clip's, two ways.

Grade (default): fit per-channel tone curves on background pixels only and print them as an ffmpeg `curves` filter for the
whole render. The person mask from motion_guide (masks.npz) keeps the person out of the fit, since a swapped person (new
hair, new skin) is not supposed to match the source.

Inside a mask (--inside face): every frame, move the render's colour inside that mask (a masks.npz key) to the source's,
by the mean red/yellow difference; brightness stays the render's. Writes the matched render to --out.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from cv_tools.media import get_fps, read_frames

POINTS = 17
SMOOTH = 0.5   # seconds in the moving median of the per-frame shift: a light change passes, flicker does not
FEATHER = 41   # px of blur on the mask's edge, at render size
ERODE = 15     # px the mask shrinks before measuring, so hair and background at its edge stay out of the means
EXACT = "scale=flags=accurate_rnd+full_chroma_int"


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


def grade(args):
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


def size_of(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                                   "-of", "csv=p=0", str(path)], text=True)
    return tuple(map(int, out.strip().split(",")[:2]))


def exact_frames(path):
    # [Agent] The matched render is decoded and encoded again, so the round trip must be exact: ffmpeg's (and OpenCV's) default
    # [Agent] YUV<->RGB rounding darkened every pixel by about 2 levels, hair and background included.
    w, h = size_of(path)
    ffmpeg = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf", EXACT, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                              stdout=subprocess.PIPE)
    # [Agent] The source is often longer than the render and is not read to its end: kill its decoder instead of letting it
    # [Agent] die on a closed pipe and print a broken-pipe error.
    try:
        while chunk := ffmpeg.stdout.read(w * h * 3):
            yield np.frombuffer(chunk, np.uint8).reshape(h, w, 3)
    finally:
        ffmpeg.kill()
        ffmpeg.wait()


def lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def colourful(lab_image):
    # [Agent] Only colourful pixels are measured and moved: skin is, grey or ash hair, brows, lashes and eye whites are not.
    # [Agent] Chroma under 10 weighs 0, over 20 weighs 1.
    return np.clip((np.hypot(lab_image[..., 1], lab_image[..., 2]) - 10) / 10, 0, 1)


def match_inside(args):
    region = np.load(args.masks)
    if args.inside not in region.files:
        raise SystemExit(f"{args.masks} has {', '.join(region.files)}, not {args.inside}; motion_guide --color saves face and hair")
    region = region[args.inside]
    source_fps, render_fps = get_fps(args.source), get_fps(args.render)
    w, h = size_of(args.render)

    def mask_at(i):
        # [Agent] The render may run at another frame rate: render frame i shows the source at time i / render_fps.
        k = min(round(i / render_fps * source_fps), len(region) - 1)
        return k, cv2.resize(region[k].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    shifts = []
    source = exact_frames(args.source)
    k_source, src = 0, next(source)
    for i, ren in enumerate(exact_frames(args.render)):
        k, mask = mask_at(i)
        while k_source < k and (frame := next(source, None)) is not None:
            k_source, src = k_source + 1, frame
        core = cv2.erode(mask, np.ones((ERODE, ERODE), np.uint8)) > 0
        src_lab, ren_lab = lab(cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)), lab(ren)
        src_w, ren_w = colourful(src_lab) * core, colourful(ren_lab) * core
        if src_w.sum() < 200 or ren_w.sum() < 200:
            shifts.append((np.nan, np.nan))
            continue
        shifts.append([(src_lab[..., c] * src_w).sum() / src_w.sum() - (ren_lab[..., c] * ren_w).sum() / ren_w.sum() for c in (1, 2)])
    shifts = np.array(shifts)
    found = ~np.isnan(shifts[:, 0])
    if not found.any():
        raise SystemExit(f"color_match: the {args.inside} mask is empty on every frame")

    # [Agent] Frames too small to measure take the nearest measured shift, then a moving median holds it steady for SMOOTH s.
    index = np.arange(len(shifts))
    shifts = np.stack([np.interp(index, index[found], shifts[found, c]) for c in range(2)], 1)
    half = round(SMOOTH * render_fps) // 2
    shifts = args.strength * np.array([np.median(shifts[max(0, i - half):i + half + 1], 0) for i in index])
    print(f"color_match: {args.inside} measured on {found.sum()}/{len(shifts)} frames, shift red {shifts[:, 0].min():+.1f}.."
          f"{shifts[:, 0].max():+.1f}, yellow {shifts[:, 1].min():+.1f}..{shifts[:, 1].max():+.1f} (Lab a*, b*)", file=sys.stderr)

    ffmpeg = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(render_fps),
                               "-i", "-", "-i", str(args.render), "-map", "0:v", "-map", "1:a?",
                               "-vf", f"{EXACT}:out_color_matrix=bt709:out_range=tv", "-colorspace", "bt709", "-color_primaries", "bt709",
                               "-color_trc", "bt709", "-color_range", "tv", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
                               "-c:a", "copy", str(args.out)], stdin=subprocess.PIPE)
    for i, ren in enumerate(exact_frames(args.render)):
        mask = mask_at(i)[1]
        if mask.any():
            ren_lab = lab(ren)
            weight = cv2.GaussianBlur(mask.astype(np.float32), (FEATHER, FEATHER), 0) * colourful(ren_lab)
            ren_lab[..., 1] += weight * shifts[i, 0]
            ren_lab[..., 2] += weight * shifts[i, 1]
            # [Agent] Pixels the weight does not touch keep their exact bytes; the Lab round trip would round them off by a level.
            moved = np.clip(cv2.cvtColor(ren_lab, cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)
            ren = np.where(weight[..., None] > 0, moved, ren)
        ffmpeg.stdin.write(np.ascontiguousarray(ren).tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise RuntimeError(f"ffmpeg failed writing {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="The clip the render was made from")
    p.add_argument("--render", type=Path, required=True, help="The render to match, same timing as --source")
    p.add_argument("--masks", type=Path, help="motion_guide's masks.npz for --source; default: every pixel counts")
    p.add_argument("--inside", help="Match inside this masks.npz mask (face, hair: motion_guide --color) and write --out")
    p.add_argument("--out", type=Path, help="With --inside: the matched render (.mp4, the render's sound copied)")
    p.add_argument("--start", type=float, default=0.0, help="First second sampled (default 0)")
    p.add_argument("--end", type=float, help="Last second sampled; leave out dissolves and fades (grade only, required there)")
    p.add_argument("--step", type=float, default=0.25, help="Seconds between samples (default 0.25)")
    p.add_argument("--width", type=int, default=320, help="Sample width in pixels (default 320)")
    p.add_argument("--grow", type=int, default=18, help="Pixels the person mask grows at sample width (default 18)")
    p.add_argument("--strength", type=float, default=1.0, help="1 = full match, 0.5 = halfway back to no change (default 1)")
    args = p.parse_args()
    if args.inside:
        if not (args.masks and args.out):
            p.error("--inside needs --masks and --out")
        match_inside(args)
    elif args.end is None:
        p.error("--end is required for the grade")
    else:
        grade(args)


if __name__ == "__main__":
    main()
