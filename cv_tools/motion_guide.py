"""Motion guide: a clip in, the guide video for an H3 character swap out (the performer blurred inside her SAM 3.1 mask,
her pose drawn on top)"""
import argparse
import contextlib
import json
import math
import os
import subprocess
import time
import types
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2.functional import gaussian_blur

from cv_tools.cli import parse
from cv_tools.media import get_fps
from cv_tools.sam31 import CHECKPOINT, load_tracker, read_video, track
from cv_tools.sapiens2.pose import OPENPOSE_BODY, RESOLUTIONS, Pose, pose_engine, pose_model

# [Agent] The parts --include / --exclude pick, as DWPose draws them. A joint shared by two parts (a shoulder, a hip) is drawn
# [Agent] when either part is in; a limb is drawn only when both its joints are. hands and face are their own point sets.
BODY_PARTS = {"head": ["nose", "right_eye", "left_eye", "right_ear", "left_ear"],
              "torso": ["neck", "right_shoulder", "left_shoulder", "right_hip", "left_hip"],
              "arms": ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist"],
              "legs": ["right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]}
PARTS = ["head", "face", "torso", "arms", "hands", "legs"]
# [Agent] --color, approved 2026-11-23: a BGR tint per label_regions label (0 none, 1 face green, 2 hair red, 3 body blue),
# [Agent] mixed 55 % into the blur so H3 can tell her parts apart where the blur hides them.
TINT = {"colors": [(0, 0, 0), (60, 190, 60), (40, 40, 210), (200, 90, 30)], "alpha": 0.55}


def is_dwpose_face_point(name):
    """Sapiens face points kept to match DWPose's eyebrows + mouth: each brow's upper contour (5), the outer lip ring (15),
    the inner lip's corners and centres (4). 29 dots, DWPose draws 30; Sapiens' denser clusters get painted by H3."""
    if "eyebrow" in name:
        return name.startswith(("upper_", "end_"))
    if "lip" in name or "mouth" in name:
        return "outer" in name or "corner" in name or name.startswith("center_")
    return False


def load_pose(threshold, face, resolution, model):
    """Sapiens2 pose `model` (a checkpoint folder; TensorRT FP16) at `resolution` (full | half), drawing DWPose-style with the chosen face dots."""
    h, w = RESOLUTIONS[resolution]
    engine, _ = pose_engine(model, (h, w))
    # [Agent] Pose takes the sapiens2 CLI's parsed arguments; these are the fields it reads.
    pose = Pose(types.SimpleNamespace(model=model, engine=engine, device="cuda", resolution=(h, w), threshold=threshold,
                                      landmarks=["all"], style="dwpose", box=None))
    pose.openpose_face = {"all": pose.openpose_face, "none": [],
                          "brows_mouth": [i for i in pose.openpose_face if is_dwpose_face_point(pose.names[i])]}[face]
    return pose


def part_points(pose, parts):
    """Bool [points]: the keypoints the drawing keeps for these parts."""
    body = dict(zip(OPENPOSE_BODY, pose.openpose_body))
    points = {"face": pose.openpose_face, "hands": [i for hand in pose.openpose_hands for i in hand]}
    keep = np.zeros(len(pose.names), bool)
    for part in parts:
        ids = points[part] if part in points else [body[n] for n in BODY_PARTS[part]]
        keep[[i for i in ids if i is not None]] = True
    return keep


def parts_list(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts or set(parts) - set(PARTS):
        raise argparse.ArgumentTypeError(f"parts are {', '.join(PARTS)}, comma-separated")
    return parts


def predict_person(pose, image, box=None):
    """(points [308, 2], scores [308]) of the person in box (x, y, w, h); the whole frame when None."""
    points, scores, _ = pose.predict(image, box)
    # [Agent] DWPose has no neck landmark and draws it at the shoulders' midpoint; Sapiens' neck sits higher. Match DWPose.
    neck, right, left = (pose.openpose_body[i] for i in (1, 2, 5))
    points[neck] = (points[right] + points[left]) / 2
    scores[neck] = min(scores[right], scores[left])
    return points, scores


def shot_starts(frames, ratio=3.0, min_change=15.0, window=2, min_length=15):
    """First frame of every shot, 0 included; frames: BGR uint8 [frames, H, W, 3]. A hard cut is PySceneDetect's
    AdaptiveDetector at its defaults: the frame's mean HSV change from the one before is at least `min_change` levels and
    `ratio` times the average change of the `window` frames on each side, and the shot before it is `min_length` frames or more."""
    h, w = frames.shape[1:3]
    scale = max(1, w // 256)  # PySceneDetect's working width: ~256 px
    small = torch.cat([F.interpolate(frames[i:i + 64].permute(0, 3, 1, 2).float(), size=(h // scale, w // scale), mode="area")
                       for i in range(0, len(frames), 64)]).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in small]).astype(np.float32)
    change = np.r_[0, np.abs(np.diff(hsv, axis=0)).mean(axis=(1, 2, 3))]
    starts = [0]
    # [Agent] Against the neighbours, not a fixed level: a whip pan changes every frame a lot, a cut changes one frame far more than the ones around it.
    for i in range(window, len(change) - window):
        around = np.r_[change[i - window:i], change[i + 1:i + window + 1]].mean()
        if change[i] >= min_change and change[i] >= ratio * around and i - starts[-1] >= min_length:
            starts.append(i)
    return starts


def find_people(pose, frames, candidates, threshold):  # frames: BGR uint8 [frames, H, W, 3] on the GPU
    """Yields (frame index, her body joints as SAM clicks) for each candidate frame where a person appears, in order."""
    # [Agent] The first frame with a person, not the "clearest" one: ranking by joints x body area picked frame 84 of a
    # selfie that opens on her face, and SAM lost her tracking back through the camera swing (frames 0-34 unmasked).
    for index in candidates:
        points, scores = predict_person(pose, frames[index])
        visible = points[[i for i in pose.openpose_body if scores[i] >= threshold and np.isfinite(points[i]).all()]]
        if len(visible) >= 4:
            yield index, [tuple(map(float, p)) for p in visible]


def text_model():
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    # [Agent] The SAM 3.1 checkpoint carries the text-prompted detector too ("detector.*" keys); Meta's builder loads it
    # from the file we have. Only the RoPE buffers come up "missing": they are rebuilt at init, not learned.
    return Sam3Processor(build_sam3_image_model(checkpoint_path=str(CHECKPOINT), load_from_HF=False))


def text_masks(processor, frame, clicks, texts):
    """SAM 3's mask for each text on one frame (BGR uint8 on the GPU): the union of the instances under her joint clicks,
    or None (always None for a None text)."""
    found = []
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        state = processor.set_image(Image.fromarray(frame.flip(-1).cpu().numpy()))
        for text in texts:
            if text is None:
                found.append(None)
                continue
            # [Agent] SAM 3 grounds one noun phrase per call, so "head, hair" runs each phrase on the same image features and
            # [Agent] pools every instance found. Every text reuses those features: one image pass per frame.
            masks = np.concatenate([processor.set_text_prompt(prompt=phrase.strip(), state=state)["masks"].squeeze(1).float().cpu().numpy() > 0
                                    for phrase in text.split(",") if phrase.strip()])
            # [Agent] Several people can be in frame (posters, passers-by): keep the instances her own joints fall inside. A union,
            # not the best one, because a part prompt ("face, hair") returns one instance per part. None of them under her
            # joints means another person or a blended frame, so no seed rather than a guess.
            keep = [m for m in masks if any(m[min(int(y), m.shape[0] - 1), min(int(x), m.shape[1] - 1)] for x, y in clicks)]
            found.append(np.logical_or.reduce(keep) if keep else None)
    return found


def empty_runs(masks, starts):
    """[(first, end)] of the runs of frames with an empty mask, split where a shot starts."""
    runs = []
    for i, empty in enumerate((~masks.flatten(1).any(dim=1)).tolist()):
        if empty and runs and runs[-1][1] == i and i not in starts:
            runs[-1] = (runs[-1][0], i + 1)
        elif empty:
            runs.append((i, i + 1))
    return runs


def track_person(frames, pose, starts, texts, args, timed):
    """([masks bool [frames, H, W] on the GPU] one per text, prompts {frame: clicks}). A text is the SAM 3 prompt its track
    starts from; None starts it from her joint clicks. Each shot is tracked on its own from its first frame with a person
    (and the text's mask); then every run of frames a track lost is prompted again the same way and tracked on its own,
    until no new prompt frame is left. Frames with nobody to find (black, a fade, an empty shot) stay empty."""
    masks = [torch.zeros(frames.shape[:3], dtype=torch.bool, device=frames.device) for _ in texts]
    spans = {shot: list(range(len(texts))) for shot in zip(starts, starts[1:] + [len(frames)])}
    tracker, processor, prompts, tried = None, None, {}, defaultdict(set)  # tried: frame -> texts looked for there
    while spans:
        found = {}
        for (first, end), waiting in spans.items():
            # [Agent] A frame stays a candidate until every text still waiting in this span has been looked for there.
            forced = (args.prompt_frame is not None and first <= args.prompt_frame < end
                      and not tried[args.prompt_frame] >= set(waiting))
            people = find_people(pose, frames, [args.prompt_frame] if forced else
                                 [i for i in range(first, end, args.prompt_every) if not tried[i] >= set(waiting)], args.threshold)
            while waiting and (person := timed("find_prompt", lambda: next(people, None))):
                frame, clicks = person
                asked = [t for t in waiting if t not in tried[frame]]
                tried[frame].update(asked)
                seeds = [None] * len(asked)
                if any(texts[t] for t in asked):
                    if processor is None:
                        processor = timed("mask_prompt", text_model)
                    seeds = timed("mask_prompt", lambda: text_masks(processor, frames[frame], clicks, [texts[t] for t in asked]))
                # [Agent] A track with a text must start from that text's mask: tracking her clicks instead turned a face
                # prompt into a whole-body mask on a dissolve frame. So a frame without one passes to the next candidate.
                for t, seed in zip(asked, seeds):
                    if seed is not None or texts[t] is None:
                        found.setdefault((first, end, frame), []).append((t, seed))
                        prompts[frame] = clicks
                        waiting.remove(t)
        if not found:
            break
        if tracker is None:
            # [Agent] SAM 3 is freed before SAM 3.1 loads, so on the first round the two never share the card; re-prompt rounds keep both.
            processor = None
            torch.cuda.empty_cache()
            tracker = timed("load_tracker", load_tracker)
        for (first, end, frame), seeded in found.items():
            # [Agent] A lost run is tracked alone, so the frames the earlier track kept are never touched. The texts seeded
            # [Agent] on one frame are tracked in one pass.
            tracked = timed("track", lambda: track(tracker, frames[first:end], frame - first,
                                                   [{"points": prompts[frame], "mask": seed} for _, seed in seeded], stride=args.stride)[0])
            for (t, _), mask in zip(seeded, tracked):
                masks[t][first:end] = mask
        spans = {}
        for t, mask in enumerate(masks):
            for run in empty_runs(mask, starts):
                spans.setdefault(run, []).append(t)
    return masks, prompts


def label_regions(face, hair, parts):
    """uint8 [frames, H, W]: each pixel's TINT row, face over hair over the rest of her (body). A region is labelled only
    when its part is drawn (face: face, head: hair, torso/arms/hands/legs: body); 0 leaves it untinted."""
    body = bool(set(parts) & {"torso", "arms", "hands", "legs"})
    labels = torch.full(face.shape, 3 if body else 0, dtype=torch.uint8, device=face.device)
    labels.masked_fill_(hair, 2 if "head" in parts else 0)
    return labels.masked_fill_(face, 1 if "face" in parts else 0)


def mask_box(mask, pad=0.1):
    """(x, y, w, h) around a GPU mask, grown by `pad` of its size on each side, clipped to the frame."""
    h, w = mask.shape
    rows, cols = torch.nonzero(mask.any(dim=1)), torch.nonzero(mask.any(dim=0))
    y0, y1, x0, x1 = torch.cat([rows.min(0)[0], rows.max(0)[0] + 1, cols.min(0)[0], cols.max(0)[0] + 1]).tolist()
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0, x1, y1 = max(0, x0 - dx), max(0, y0 - dy), min(w, x1 + dx), min(h, y1 + dy)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def blur(images, sigma):
    """Gaussian blur of [N, C, H, W] floats; the kernel spans ±3 sigma, like OpenCV's default."""
    size = 2 * math.ceil(3 * sigma) + 1
    return gaussian_blur(images, [size, size], [sigma, sigma])


def heavy_blur(image, sigma):
    # [Agent] A large blur keeps only low frequencies, so shrinking by sigma/10 first loses nothing visible: at the default
    # [Agent] (sigma ~87 px on a 1312 px tall clip) a few ms instead of 910 ms on the CPU. A blur wider than the frame is a
    # [Agent] flat average anyway, so its sigma is capped at what fits (reflect padding must stay smaller than the frame).
    h, w = image.shape[-2:]
    shrink = max(1, int(sigma // 10))
    small = F.interpolate(image, size=(h // shrink, w // shrink), mode="area")
    sigma = min(sigma / shrink, (min(small.shape[-2:]) - 1) / 6)
    return F.interpolate(blur(small, sigma), size=(h, w), mode="bilinear", align_corners=False)


def compose(frame, mask, drawing, blur_div, mask_grow=0.01, labels=None):
    """GPU: the frame blurred inside the mask (grown by `mask_grow` of the long side in total, 2 px feather), the blur
    tinted by `labels` (label_regions' frame) when given, the drawing on top. uint8 [H, W, 3]."""
    h, w = mask.shape
    image = frame.permute(2, 0, 1)[None].float()
    grow = max(3, int(mask_grow * max(w, h)) | 1)
    # [Agent] A max-pool with stride 1 is a square dilation, OpenCV's dilate with a ones kernel.
    alpha = blur(F.max_pool2d(mask[None, None].float(), grow, stride=1, padding=grow // 2), 2.0)
    blurred = heavy_blur(image, max(w, h) / blur_div)
    if labels is not None:
        # [Agent] The tint goes into the blur, not onto the finished frame: it stays inside the blurred area, fades with
        # [Agent] its feather, and the drawing pasted after it keeps its own colours.
        tint = torch.tensor(TINT["colors"], dtype=torch.float32, device=labels.device)[labels.long()].permute(2, 0, 1)[None]
        weight = (labels > 0) * TINT["alpha"]
        blurred = blurred * (1 - weight) + tint * weight
    out = (image * (1 - alpha) + blurred * alpha)[0].permute(1, 2, 0)
    # [Agent] Only the drawing's bright pixels replace the frame: its dark anti-aliased edges would leave a dark fringe.
    drawn = drawing.amax(dim=2) > 30
    out[drawn] = drawing[drawn].float()
    return out.to(torch.uint8)


@contextlib.contextmanager
def h264_writer(path, w, h, fps, audio=None):
    """Yields write(frame_bgr uint8 [h, w, 3]); libx264 CRF 14, yuv420p, padded to even sides as x264 requires."""
    # [Agent] The guide carries the source clip's sound (when it has any), so a workflow loads one file for picture and audio.
    # "1:a?" skips a silent source; -shortest trims the audio to the frames actually written.
    sound = ["-i", str(audio), "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"] if audio else []
    ffmpeg = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                               "-r", str(fps), "-i", "-", *sound, "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                               "-crf", "14", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    try:
        yield ffmpeg.stdin.write
    finally:
        with contextlib.suppress(BrokenPipeError):
            ffmpeg.stdin.close()
        if ffmpeg.wait():
            raise RuntimeError(f"ffmpeg failed writing {path}")


def write_guide(path, frames, masks, pose, blur_div, fps, stream, mask_grow=0.01, audio=None, keep=None, labels=None):
    """Pose each frame inside her mask's box, then draw the `keep` points (part_points; all when None), blur (tinted by
    label_regions' `labels` when given) and encode it. frames (BGR uint8), masks and labels live on the GPU; returns
    (points, scores) per frame, every point, drawn or not."""
    total, h, w = frames.shape[:3]
    points = np.full((total, len(pose.names), 2), np.nan, np.float32)
    scores = np.zeros((total, len(pose.names)), np.float32)
    has_person = masks.flatten(1).any(dim=1).tolist()
    blank = np.zeros((h, w, 3), np.uint8)  # draw_dwpose only reads its canvas size from the frame

    def draw_and_write(index, write):
        # [Agent] Inference mode and the CUDA stream are per thread, so the worker sets both for its GPU work.
        with torch.inference_mode(), torch.cuda.stream(stream):
            if not has_person[index]:
                return write(frames[index].cpu().numpy())
            # [Agent] The skeleton is drawn on the CPU (OpenCV); the frame and mask never leave the GPU.
            drawn = scores[index] if keep is None else np.where(keep, scores[index], 0)
            drawing = torch.from_numpy(pose.draw_dwpose(blank, points[index], drawn)[0]).cuda()
            write(compose(frames[index], masks[index], drawing, blur_div, mask_grow,
                          None if labels is None else labels[index]).cpu().numpy())

    # [Agent] One worker draws, composes and encodes frame i while the GPU runs frame i+1's pose. Waiting for the previous
    # [Agent] frame before queueing the next keeps one frame in flight and raises a failure on the frame that caused it.
    with h264_writer(path, w, h, fps, audio) as write, ThreadPoolExecutor(max_workers=1) as worker:
        pending = None
        for index in range(total):
            if has_person[index]:
                # [Agent] Pose only inside her mask's box: other people in the frame never get her skeleton.
                points[index], scores[index] = predict_person(pose, frames[index], mask_box(masks[index]))
            if pending:
                pending.result()
            pending = worker.submit(draw_and_write, index, write)
        if pending:
            pending.result()
    return points, scores


@torch.inference_mode()
def run(args):
    if args.is_image or args.device != "cuda":
        raise SystemExit("motion_guide takes a video and runs on CUDA")
    if args.benchmark:
        raise SystemExit("motion_guide has no --benchmark mode; summary.json has the time of every stage")
    torch.cuda.reset_peak_memory_stats()
    seconds = {}

    def timed(stage, work):
        started = time.perf_counter()
        result = work()
        seconds[stage] = seconds.get(stage, 0) + time.perf_counter() - started
        return result

    # [Agent] TensorRT adds syncs on the default stream; one explicit stream keeps the ordering without them.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # [Agent] The clip goes to the GPU once (reel01: ~0.6 GB); SAM, pose and the composite all read it there.
        frames = timed("decode", lambda: torch.from_numpy(read_video(args.input, args.frames, args.size)).cuda())
        pose = timed("load_pose", lambda: load_pose(args.threshold, args.face, args.pose_res, args.pose_model))
        parts = args.include or [p for p in PARTS if p not in (args.exclude or [])]
        starts = timed("shots", lambda: shot_starts(frames))
        texts = [args.mask_prompt] + (["face", "hair"] if args.color else [])
        masks, prompts = track_person(frames, pose, starts, texts, args, timed)
        labels = label_regions(*masks[1:], parts) if args.color else None
        masks = masks[0]
        lost = empty_runs(masks, [])
        unmasked = sum(end - first for first, end in lost)
        if unmasked == len(frames):
            raise SystemExit("No person found in the clip")
        # [Agent] An empty mask means no blur and no pose on that frame: right on black or a fade, a leak if she is there.
        if lost:
            print(f"motion_guide: no mask on {unmasked} frames ({', '.join(f'{first}-{end - 1}' for first, end in lost)}); "
                  "they go out as in the source")
        torch.cuda.empty_cache()  # SAM's ~4 GB, freed when track_person returned, before pose and drawing
        args.out.mkdir(parents=True, exist_ok=True)
        points, scores = timed("pose_draw_encode", lambda: write_guide(
            args.out / "guide.mp4", frames, masks, pose, args.blur_div, get_fps(args.input), stream, args.mask_grow, args.input,
            part_points(pose, parts), labels))
    timed("save", lambda: (np.savez_compressed(args.out / "masks.npz", masks=masks.cpu().numpy()),
                           np.savez_compressed(args.out / "keypoints.npz", points=points, scores=scores,
                                               names=np.array(pose.names))))
    total = sum(seconds.values())
    summary = {"input": str(args.input), "frames": len(frames), "size_wh": [frames.shape[2], frames.shape[1]],
               "shot_starts": starts, "prompt_frames": list(prompts), "clicks": prompts, "mask_prompt": args.mask_prompt,
               "color": args.color, "empty_frames": [[first, end - 1] for first, end in lost],
               "stride": args.stride, "pose_model": args.pose_model.name, "pose_res": args.pose_res,
               "face_dots": args.face, "face_dot_count": len(pose.openpose_face), "parts": parts,
               "seconds": {k: round(v, 2) for k, v in seconds.items()} | {"total": round(total, 2)},
               "fps": round(len(frames) / total, 2),
               # [Agent] PyTorch's allocator only: the TensorRT pose engine's memory is not counted here.
               "peak_torch_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"motion_guide: {len(frames)} frames in {total:.1f}s ({summary['fps']} FPS), "
          f"{len(starts)} shots, prompt frames {list(prompts)} -> {args.out}")


def main():
    # [Agent] Grow the CUDA memory pool in place instead of in fixed blocks: tracking left ~1.8 GB reserved but unused.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stride", type=int, default=2, help="SAM tracks every Nth frame, the rest filled in (default 2)")
    # [Agent] 0.8b by default: same skeleton as 0.4b on full-body and close selfie clips, but it keeps the arms on low angles along
    # the body where 0.4b drops them (jeans reel, 2026-11-19); a few seconds more per guide, the render time is unchanged.
    p.add_argument("--pose-model", type=pose_model, default="0.8b",
                   help="Sapiens2 pose size (0.4b | 0.8b | 1b | 5b) or a checkpoint path; its engine builds on first use (default 0.8b)")
    p.add_argument("--pose-res", choices=["full", "half"], default="full",  # the two pose engines that are built
                   help="Pose input: full (1024x768) or half (512x384: 3.5x faster, joints a median 4 px off)")
    p.add_argument("--mask-prompt", help="Start the track from SAM 3's mask for this text instead of the joint clicks: "
                                         "'person' keeps her clothes in; 'head, hair' for a face-only swap (her body and clothes stay visible)")
    # [Agent] A wider blur hides what sticks out of the tracked mask (flyaway hair, a bun's outline, dangling straps) that H3 would otherwise rebuild.
    p.add_argument("--mask-grow", type=float, default=0.01, help="Blur area grown beyond the mask, as a fraction of the long side in total (default 0.01)")
    p.add_argument("--blur-div", type=float, default=15, help="Blur sigma = longest side / this (default 15: heavy)")
    p.add_argument("--face", choices=["brows_mouth", "all", "none"], default="brows_mouth",
                   help="Face dots: eyebrows + mouth at DWPose density (default), all Sapiens face dots, or none")
    drawn = p.add_mutually_exclusive_group()
    drawn.add_argument("--include", type=parts_list, help=f"Draw only these parts, comma-separated ({', '.join(PARTS)}); "
                       "face swap: head,face")
    drawn.add_argument("--exclude", type=parts_list, help="Draw every part but these, e.g. legs")
    p.add_argument("--color", action="store_true", help="Tint the blur per region so H3 tells her parts apart: face green, "
                   "hair red, the rest of her blue, each only when its part is drawn (face; head; torso, arms, hands or legs)")
    p.add_argument("--prompt-every", type=int, default=6, help="Frames between pose checks while looking for the first frame with a person")
    p.add_argument("--prompt-frame", type=int, help="Start SAM on this frame instead of its shot's first frame with a person")
    p.add_argument("--threshold", type=float, default=0.3, help="Keypoint score to trust and draw (default 0.3)")
    run(parse(p, "motion_guide"))


if __name__ == "__main__":
    main()
