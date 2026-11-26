"""Remove black bars (letterbox, pillarbox, a picture inside the picture) shot by shot, keeping the frame size and the sound.

    python -m cv_tools.bars --input clip.mp4 --output out.mp4

The clip is split into shots with cv_tools.shots (hard cuts), then again wherever the bars change without a hard cut, such as
a dissolve into a picture inside black bars. A shot with bars is cropped to the largest box with the frame's aspect that fits
inside its picture, kept on the person's head, and scaled back to the frame size. A shot without bars passes through.
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from cv_tools.media import get_fps
from cv_tools.shots import CUT, MIN_SHOT, scene_scores, shot_starts, stamp

SAMPLES = 16
LIT = 7  # brightest channel above this is picture, not bar: bars decode to 0-3; on 6 clips 6-8 holds, and a dark night scene splits at 10
EDGE = 0.1  # a row or column belongs to the picture when this share of its pixels is lit
BARS = 0.96  # a picture at least this wide and tall in both directions is the whole frame
MOVED = 0.05  # picture edges that move more than this share of the frame start a new shot
HEAD = [0, 1, 2, 3, 4]  # Sapiens2 nose, eyes, ears
HEAD_SCORE = 0.3
HEAD_PAD = 0.8  # the face points' span again on every side, to cover forehead, hair and chin
NO_BOX = (np.inf, np.inf, -np.inf, -np.inf)  # the empty box: any union with it gives the other box


def find_box(frame):
    # [Agent] inRange marks every pixel whose three channels are all at or below LIT in one SIMD pass (a max over the channels
    # [Agent] was 5x slower); the share of dark pixels per column and row then gives the lit share.
    dark = cv2.inRange(frame, (0, 0, 0), (LIT, LIT, LIT))
    cols = np.flatnonzero(cv2.reduce(dark, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F)[0] < 255 * (1 - EDGE))
    rows = np.flatnonzero(cv2.reduce(dark, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F)[:, 0] < 255 * (1 - EDGE))
    return (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1) if len(cols) and len(rows) else NO_BOX


def read_boxes(path):
    cap = cv2.VideoCapture(str(path))
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes = []
    while cap.grab():
        boxes.append(find_box(cap.retrieve()[1]))
    cap.release()
    return np.array(boxes, dtype=float), size


def union(boxes):
    return np.r_[boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]


def split_bars(boxes, start, end, size):
    """Frames in [start, end) where the picture changes to another one that then holds for MIN_SHOT frames."""
    tol = MOVED * np.array([size[0], size[1]] * 2)
    starts, held = [start], union(boxes[start:start + MIN_SHOT])
    for i in range(start + MIN_SHOT, end - 2 * MIN_SHOT + 1):
        # [Agent] The union of the next MIN_SHOT frames flips on the first frame whose whole window is the new picture. It must
        # [Agent] equal the window after it, so a dissolve still fading in, a flicker or a fade out never starts a shot. A hard
        # [Agent] jump back to a full frame flips MIN_SHOT frames early, so the start is the first frame that already matches it.
        ahead, later = union(boxes[i:i + MIN_SHOT]), union(boxes[i + MIN_SHOT:i + 2 * MIN_SHOT])
        if np.isfinite(ahead).all() and (np.abs(ahead - held) > tol).any() and not (np.abs(ahead - later) > tol).any():
            starts.append(next(k for k in range(i, end) if np.abs(boxes[k] - ahead).sum() < np.abs(boxes[k] - held).sum()))
            held = ahead
    return starts


def find_picture(boxes, size):
    box = union(boxes)
    return (0, 0, *size) if not np.isfinite(box).all() else (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))


def read_samples(path, indices):
    cap = cv2.VideoCapture(str(path))
    wanted, frames = set(indices), []
    for i in range(max(indices) + 1):
        cap.grab()
        if i in wanted:
            frames.append(cap.retrieve()[1])
    cap.release()
    return frames


def find_head(pose, frames, picture):
    """The box (x0, y0, x1, y1) around her head over all the samples, or None when no sample shows a person."""
    x, y, w, h = picture
    boxes = []
    for frame in frames:
        # [Agent] Pose sees the picture alone, so she fills the model's input instead of a strip between two black bars.
        points, scores, _ = pose.predict(np.ascontiguousarray(frame[y:y + h, x:x + w]))
        face = points[[i for i in HEAD if scores[i] >= HEAD_SCORE]]
        if len(face) >= 3:
            pad = HEAD_PAD * (face.max(axis=0) - face.min(axis=0)).max()
            boxes.append((*(face.min(axis=0) - pad + (x, y)), *(face.max(axis=0) + pad + (x, y))))
    return (*np.min(boxes, axis=0)[:2], *np.max(boxes, axis=0)[2:]) if boxes else None


def pick_crop(picture, size, head):
    x, y, w, h = picture
    # [Agent] Largest box with the frame's aspect that fits the picture. One box serves the whole shot, so it sits on the
    # [Agent] middle of the head's full range, which keeps the head in as many frames as one box can; no head: the picture's middle.
    scale = min(w / size[0], h / size[1])
    cw, ch = int(size[0] * scale), int(size[1] * scale)
    cx, cy = (x + w / 2, y + h / 2) if head is None else ((head[0] + head[2]) / 2, (head[1] + head[3]) / 2)
    return int(np.clip(cx - cw / 2, x, x + w - cw)) // 2 * 2, int(np.clip(cy - ch / 2, y, y + h - ch)) // 2 * 2, cw, ch


def join_filter(shots, size):
    parts = []
    for n, (a, b, crop) in enumerate(shots):
        cut = f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]},scale={size[0]}:{size[1]}:flags=lanczos," if crop else ""
        parts.append(f"[0:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS,{cut}setsar=1[s{n}]")
    return ";".join(parts) + ";" + "".join(f"[s{n}]" for n in range(len(shots))) + f"concat=n={len(shots)}:v=1:a=0[v]"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    fps = get_fps(args.input)
    boxes, size = read_boxes(args.input)
    cuts = shot_starts(scene_scores(args.input), CUT, MIN_SHOT)
    shots, pose = [], None
    for a, b in zip(cuts, cuts[1:] + [len(boxes)]):
        starts = split_bars(boxes, a, b, size)
        for s, e in zip(starts, starts[1:] + [b]):
            picture, crop = find_picture(boxes[s:e], size), None
            if picture[2] < BARS * size[0] or picture[3] < BARS * size[1]:
                if pose is None:
                    # [Agent] Imported here so a clip without bars never pays for torch and the TensorRT engine.
                    from cv_tools.motion_guide import load_pose
                    from cv_tools.sapiens2.pose import pose_model
                    pose = load_pose(HEAD_SCORE, "none", "half", pose_model("0.8b"))
                frames = read_samples(args.input, np.linspace(s, e - 1, min(SAMPLES, e - s)).astype(int))
                crop = pick_crop(picture, size, find_head(pose, frames, picture))
            shots.append((s, e, crop))
            print(f"shot {len(shots)}  {stamp(s / fps)} - {stamp(e / fps)}  picture {picture[2]}x{picture[3]} at {picture[0]},{picture[1]}  "
                  + (f"crop {crop[2]}x{crop[3]} at {crop[0]},{crop[1]}" if crop else "no bars"))

    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(args.input), "-filter_complex", join_filter(shots, size), "-map", "[v]", "-map", "0:a?",
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "copy", str(args.output)], check=True, stdin=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
