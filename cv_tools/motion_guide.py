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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2.functional import gaussian_blur

from cv_tools.cli import parse
from cv_tools.media import get_fps
from cv_tools.sam31 import CHECKPOINT, load_tracker, read_video, track
from cv_tools.sapiens2.pose import RESOLUTIONS, Pose, pose_engine, pose_model


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


def predict_person(pose, image, box=None):
    """(points [308, 2], scores [308]) of the person in box (x, y, w, h); the whole frame when None."""
    points, scores, _ = pose.predict(image, box)
    # [Agent] DWPose has no neck landmark and draws it at the shoulders' midpoint; Sapiens' neck sits higher. Match DWPose.
    neck, right, left = (pose.openpose_body[i] for i in (1, 2, 5))
    points[neck] = (points[right] + points[left]) / 2
    scores[neck] = min(scores[right], scores[left])
    return points, scores


def find_prompt(pose, frames, candidates, threshold):  # frames: BGR uint8 [frames, H, W, 3] on the GPU
    """(frame index, her body joints as SAM clicks) on the first candidate frame where a person appears."""
    # [Agent] The first frame with a person, not the "clearest" one: ranking by joints x body area picked frame 84 of a
    # selfie that opens on her face, and SAM lost her tracking back through the camera swing (frames 0-34 unmasked).
    for index in candidates:
        points, scores = predict_person(pose, frames[index])
        visible = points[[i for i in pose.openpose_body if scores[i] >= threshold and np.isfinite(points[i]).all()]]
        if len(visible) >= 4:
            return index, [tuple(map(float, p)) for p in visible]
    raise ValueError("No person found in the clip")


def text_mask(frame, clicks, prompt):
    """SAM 3's mask for `prompt` on one frame (BGR uint8 on the GPU): the union of the instances under her joint clicks, or None."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    # [Agent] The SAM 3.1 checkpoint carries the text-prompted detector too ("detector.*" keys); Meta's builder loads it
    # from the file we have. Only the RoPE buffers come up "missing": they are rebuilt at init, not learned.
    model = build_sam3_image_model(checkpoint_path=str(CHECKPOINT), load_from_HF=False)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        processor = Sam3Processor(model)
        state = processor.set_image(Image.fromarray(frame.flip(-1).cpu().numpy()))
        # [Agent] SAM 3 grounds one noun phrase per call, so "head, hair" runs each phrase on the same image features and pools every instance found.
        masks = np.concatenate([processor.set_text_prompt(prompt=phrase.strip(), state=state)["masks"].squeeze(1).float().cpu().numpy() > 0
                                for phrase in prompt.split(",") if phrase.strip()])
    del model, processor, state
    torch.cuda.empty_cache()
    if not len(masks):
        return None
    # [Agent] Several people can be in frame (posters, passers-by): keep the instances her own joints fall inside. A union,
    # not the best one, because a part prompt ("face, hair") returns one instance per part.
    hits = [sum(bool(m[min(int(y), m.shape[0] - 1), min(int(x), m.shape[1] - 1)]) for x, y in clicks) for m in masks]
    keep = [m for m, h in zip(masks, hits) if h] or [masks[int(np.argmax(hits))]]
    return np.logical_or.reduce(keep)


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


def compose(frame, mask, drawing, blur_div, mask_grow=0.01):
    """GPU: the frame blurred inside the mask (grown by `mask_grow` of the long side in total, 2 px feather), the drawing on top. uint8 [H, W, 3]."""
    h, w = mask.shape
    image = frame.permute(2, 0, 1)[None].float()
    grow = max(3, int(mask_grow * max(w, h)) | 1)
    # [Agent] A max-pool with stride 1 is a square dilation, OpenCV's dilate with a ones kernel.
    alpha = blur(F.max_pool2d(mask[None, None].float(), grow, stride=1, padding=grow // 2), 2.0)
    out = (image * (1 - alpha) + heavy_blur(image, max(w, h) / blur_div) * alpha)[0].permute(1, 2, 0)
    # [Agent] Only the drawing's bright pixels replace the frame: its dark anti-aliased edges would leave a dark fringe.
    drawn = drawing.amax(dim=2) > 30
    out[drawn] = drawing[drawn].float()
    return out.to(torch.uint8)


@contextlib.contextmanager
def h264_writer(path, w, h, fps):
    """Yields write(frame_bgr uint8 [h, w, 3]); libx264 CRF 14, yuv420p, padded to even sides as x264 requires."""
    ffmpeg = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                               "-r", str(fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                               "-crf", "14", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    try:
        yield ffmpeg.stdin.write
    finally:
        with contextlib.suppress(BrokenPipeError):
            ffmpeg.stdin.close()
        if ffmpeg.wait():
            raise RuntimeError(f"ffmpeg failed writing {path}")


def write_guide(path, frames, masks, pose, blur_div, fps, stream, mask_grow=0.01):
    """Pose each frame inside her mask's box, then draw, blur and encode it. frames (BGR uint8) and masks live on the GPU;
    returns (points, scores) per frame."""
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
            drawing = torch.from_numpy(pose.draw_dwpose(blank, points[index], scores[index])[0]).cuda()
            write(compose(frames[index], masks[index], drawing, blur_div, mask_grow).cpu().numpy())

    # [Agent] One worker draws, composes and encodes frame i while the GPU runs frame i+1's pose. Waiting for the previous
    # [Agent] frame before queueing the next keeps one frame in flight and raises a failure on the frame that caused it.
    with h264_writer(path, w, h, fps) as write, ThreadPoolExecutor(max_workers=1) as worker:
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
        seconds[stage] = time.perf_counter() - started
        return result

    # [Agent] TensorRT adds syncs on the default stream; one explicit stream keeps the ordering without them.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # [Agent] The clip goes to the GPU once (reel01: ~0.6 GB); SAM, pose and the composite all read it there.
        frames = timed("decode", lambda: torch.from_numpy(read_video(args.input, args.frames, args.size)).cuda())
        pose = timed("load_pose", lambda: load_pose(args.threshold, args.face, args.pose_res, args.pose_model))
        candidates = [args.prompt_frame] if args.prompt_frame is not None else range(0, len(frames), args.prompt_every)
        prompt_frame, clicks = timed("find_prompt", lambda: find_prompt(pose, frames, candidates, args.threshold))
        # [Agent] SAM 3 runs and is freed before SAM 3.1 loads, so the two never share the card.
        seed = timed("mask_prompt", lambda: text_mask(frames[prompt_frame], clicks, args.mask_prompt)) if args.mask_prompt else None
        tracker = timed("load_tracker", load_tracker)
        masks, _ = timed("track", lambda: track(tracker, frames, prompt_frame, points=clicks, mask=seed, stride=args.stride))
        # [Agent] An empty mask means no blur and no pose on that frame; the guide would show the source person. Stop instead.
        empty = (masks.flatten(1).sum(1) == 0).nonzero().flatten().tolist()
        if empty:
            raise SystemExit(f"SAM lost the person on {len(empty)} frames ({empty[0]}-{empty[-1]}); "
                             f"started on frame {prompt_frame}. Pass --prompt-frame with a frame where the person is visible.")
        tracker = None  # frees SAM's ~4 GB before pose and drawing
        torch.cuda.empty_cache()
        args.out.mkdir(parents=True, exist_ok=True)
        points, scores = timed("pose_draw_encode", lambda: write_guide(
            args.out / "guide.mp4", frames, masks, pose, args.blur_div, get_fps(args.input), stream, args.mask_grow))
    timed("save", lambda: (np.savez_compressed(args.out / "masks.npz", masks=masks.cpu().numpy()),
                           np.savez_compressed(args.out / "keypoints.npz", points=points, scores=scores,
                                               names=np.array(pose.names))))
    total = sum(seconds.values())
    summary = {"input": str(args.input), "frames": len(frames), "size_wh": [frames.shape[2], frames.shape[1]],
               "prompt_frame": prompt_frame, "clicks": clicks, "mask_prompt": args.mask_prompt if seed is not None else None,
               "stride": args.stride, "pose_model": args.pose_model.name, "pose_res": args.pose_res,
               "face_dots": args.face, "face_dot_count": len(pose.openpose_face),
               "seconds": {k: round(v, 2) for k, v in seconds.items()} | {"total": round(total, 2)},
               "fps": round(len(frames) / total, 2),
               # [Agent] PyTorch's allocator only: the TensorRT pose engine's memory is not counted here.
               "peak_torch_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"motion_guide: {len(frames)} frames in {total:.1f}s ({summary['fps']} FPS), "
          f"prompt frame {prompt_frame} -> {args.out}")


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
    p.add_argument("--prompt-every", type=int, default=6, help="Frames between pose checks while looking for the first frame with a person")
    p.add_argument("--prompt-frame", type=int, help="Start SAM on this frame instead of the first frame with a person")
    p.add_argument("--threshold", type=float, default=0.3, help="Keypoint score to trust and draw (default 0.3)")
    run(parse(p, "motion_guide"))


if __name__ == "__main__":
    main()
