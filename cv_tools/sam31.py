"""SAM 3.1 (Meta, tracker only): track one object through a video from clicks or a mask on one frame"""
import argparse
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cv_tools import MODEL_DIR
from cv_tools.cli import measure, parse, write_summary
from cv_tools.media import Frame, get_fps, read_frames, write_run

CHECKPOINT = MODEL_DIR / "sam3.1" / "sam3.1_multiplex.pt"
DTYPE = torch.float16
COLOR = (255, 140, 30)  # BGR, the overlay tint
MATRIX_LAYERS = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose2d)


@contextmanager
def skip_weight_init():
    # [Agent] Random init of 876M parameters costs 1.4 s, and the checkpoint overwrites every one of them.
    layers = MATRIX_LAYERS + (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.GroupNorm)
    resets = {layer: layer.reset_parameters for layer in layers}
    inits = {name: getattr(torch.nn.init, name) for name in dir(torch.nn.init) if name.endswith("_") and name[0] != "_"}
    for layer in layers:
        layer.reset_parameters = lambda self: None
    for name in inits:
        setattr(torch.nn.init, name, lambda tensor, *args, **kwargs: tensor)
    try:
        yield
    finally:
        for layer, reset in resets.items():
            layer.reset_parameters = reset
        for name, init in inits.items():
            setattr(torch.nn.init, name, init)


def load_tracker(checkpoint=CHECKPOINT, compile_encoder=False):
    from sam3.model_builder import build_sam3_multiplex_video_model

    # [Agent] Meta's tracker-only model: their default video predictor also runs the text detector on every frame,
    # [Agent] 2x slower, and loses objects prompted only by clicks. FlashAttention 3 exists only on Hopper (sm_90).
    with skip_weight_init():
        model = build_sam3_multiplex_video_model(checkpoint_path=None, load_from_HF=False, use_rope_real=True,
                                                 use_fa3=torch.cuda.get_device_capability() == (9, 0), device="cuda")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # [Agent] The released file holds detector + tracker: keep the tracker's weights and the shared image encoder, renamed
    # [Agent] to what the tracker expects. RoPE tables (freqs_cis) are rebuilt at init, so the file does not have them.
    renames = {"tracker.model.": "", "detector.backbone.vision_backbone.": "backbone.vision_backbone."}
    tracker_weights = {name.replace(old, new, 1): tensor for name, tensor in weights.get("model", weights).items()
                       for old, new in renames.items() if name.startswith(old)}
    missing = [k for k in model.load_state_dict(tracker_weights, strict=False).missing_keys if "freqs_cis" not in k]
    if missing:
        raise ValueError(f"Checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
    # [Agent] Half precision on the matrix layers only: casting the whole model also rounds the RoPE tables and norms,
    # [Agent] which dropped mask IoU against fp32 from 0.99 to 0.95.
    for module in model.modules():
        if isinstance(module, MATRIX_LAYERS):
            module.to(DTYPE)
    if compile_encoder:
        # [Agent] Encoder only: compiling the tracker heads with CUDA graphs overwrites the outputs they keep as memory.
        vision = model.backbone.vision_backbone
        vision.trunk = torch.compile(vision.trunk, mode="max-autotune-no-cudagraphs", dynamic=False)
    model.compiled_encoder = compile_encoder
    return model.eval()


def read_video(path, limit=0, size=None):
    """The clip decoded once: BGR uint8 [frames, H, W, 3]."""
    return np.stack([frame.image for frame in read_frames(path, limit, size=size)])


def encode(model, frames):
    """(pixels, encoder outputs) for BGR uint8 GPU frames [N, H, W, 3], in one batched pass."""
    from sam3.model.data_misc import NestedTensor

    pixels = frames.permute(0, 3, 1, 2).flip(1).float()  # BGR -> RGB
    pixels = F.interpolate(pixels, size=(model.image_size,) * 2, mode="bilinear", align_corners=False)
    pixels = ((pixels / 255 - 0.5) / 0.5).to(DTYPE)
    with torch.autocast("cuda", dtype=DTYPE):
        # [Agent] need_sam3_out=False skips the detector's neck: tracking only reads the other two outputs.
        return pixels, model.forward_image(NestedTensor(tensors=pixels, mask=None), need_sam3_out=False,
                                           need_interactive_out=True, need_propagation_out=True)


def frame_of(outputs, i):
    """Frame i of a batched encoder output; every tensor in it, position encodings included, is batched."""
    from sam3.model.data_misc import NestedTensor

    if isinstance(outputs, torch.Tensor):
        return outputs[i:i + 1]
    if isinstance(outputs, NestedTensor):
        return NestedTensor(tensors=outputs.tensors[i:i + 1], mask=None if outputs.mask is None else outputs.mask[i:i + 1])
    if isinstance(outputs, dict):
        return {k: frame_of(v, i) for k, v in outputs.items()}
    if isinstance(outputs, (list, tuple)):
        return type(outputs)(frame_of(v, i) for v in outputs)
    return outputs


def batched_encoder(model, frames, batch, first_step):
    """Stand-in for the tracker's per-frame encoder call: encodes `batch` frames at once, ahead in the direction it goes."""
    cache, previous = {}, None

    def get_image_feature(inference_state, frame_idx, batch_size):  # the tracker's signature; one object, so batch_size 1
        nonlocal cache, previous
        step = first_step if previous is None else (-1 if frame_idx < previous else 1)
        previous = frame_idx
        if frame_idx not in cache:
            cache = {k: v for k, v in cache.items() if (k - frame_idx) * step >= 0}  # drop frames already passed
            ahead = [i for i in range(frame_idx, frame_idx + step * batch, step) if 0 <= i < len(frames) and i not in cache]
            # [Agent] A compiled encoder recompiles on every new batch size, so its batches are padded to one shape.
            indices = ahead + [ahead[-1]] * (batch - len(ahead)) if model.compiled_encoder else ahead
            pixels, outputs = encode(model, frames[indices])
            for j, i in enumerate(ahead):
                cache[i] = (pixels[j:j + 1], frame_of(outputs, j))
        pixels, outputs = cache[frame_idx]
        return pixels, model._prepare_backbone_features(outputs)

    return get_image_feature


def covered_range(direction, prompt_frame, total):
    """(first, last) frame that tracking reaches from the prompt frame."""
    return (prompt_frame if direction == "forward" else 0, prompt_frame if direction == "backward" else total - 1)


def add_prompt(model, state, frame, points=(), negative=(), mask=None):
    """Prompts the state's one object on `frame`: by a bool mask [H, W] when given, else by pixel clicks."""
    if mask is not None:
        model.add_new_masks(state, frame_idx=frame, obj_ids=[1], masks=torch.as_tensor(mask, device="cuda")[None])
    else:
        h, w = state["video_height"], state["video_width"]
        clicks = torch.tensor([[x / w, y / h] for x, y in [*points, *negative]], dtype=torch.float32)
        labels = torch.tensor([1] * len(points) + [0] * len(negative), dtype=torch.int32)
        model.add_new_points(state, frame_idx=frame, obj_id=1, points=clicks, labels=labels, clear_old_points=True)


@torch.inference_mode()
def track(model, frames, prompt_frame, prompts, direction="both", stride=2, batch=8):
    """(masks bool [objects, frames, H, W] on the GPU, tracking seconds): one object per prompt (add_prompt's points,
    negative or mask), all prompted on `prompt_frame`. frames: BGR uint8 [frames, H, W, 3] on the GPU. With stride N every
    Nth frame is tracked and each frame between gets the union of its two tracked neighbours."""
    total, h, w = frames.shape[:3]
    if not 0 <= prompt_frame < total:
        raise ValueError(f"Prompt frame {prompt_frame} is outside the clip's {total} frames")
    offset, start = prompt_frame % stride, prompt_frame // stride  # the tracked frames include the prompt frame
    tracked = frames[offset::stride]
    model._get_image_feature = batched_encoder(model, tracked, batch, first_step=-1 if direction == "backward" else 1)
    try:
        # [Agent] One state per object: SAM 3.1 keeps the objects of one state apart pixel by pixel, which emptied a face
        # [Agent] tracked inside its person. The states step through the clip together, so each frame's cached encoder
        # [Agent] output (~80 % of a pass) serves all of them, and every mask is bit-identical to tracking it alone.
        states = [model.init_state(video_height=h, video_width=w, num_frames=len(tracked)) for _ in prompts]
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=DTYPE):
            for state, prompt in zip(states, prompts):
                add_prompt(model, state, start, **prompt)
                model.propagate_in_video_preflight(state, run_mem_encoder=True)
            masks = {}
            for reverse in {"both": (False, True), "forward": (False,), "backward": (True,)}[direction]:
                for steps in zip(*(model.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=None,
                                                             reverse=reverse, tqdm_disable=True) for state in states)):
                    masks[offset + steps[0][0] * stride] = torch.stack([video_masks[0, 0] > 0 for _, _, _, video_masks, *_ in steps])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    finally:
        del model._get_image_feature  # restores Meta's method and breaks the closure's reference to the model
    out = torch.zeros((len(prompts), total, h, w), dtype=torch.bool, device="cuda")
    known = sorted(masks)
    first, last = covered_range(direction, prompt_frame, total)
    for f in range(first, last + 1):
        # [Agent] The union over-covers a fast move rather than cutting a limb off: the safe failure for a blur mask.
        i = np.searchsorted(known, f)
        out[:, f] = masks[f] if f in masks else masks[known[max(i - 1, 0)]] | masks[known[min(i, len(known) - 1)]]
    return out, seconds


def encode_rle(mask):
    # [Agent] COCO uncompressed RLE: run lengths over the column-major mask, starting with a background run,
    # [Agent] so pycocotools decodes it losslessly. The MP4 masks are lossy and only for viewing.
    flat = mask.ravel(order="F")
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate([[0], edges, [flat.size]]))
    if flat[0]:
        counts = np.concatenate([[0], counts])
    return {"size": list(mask.shape), "counts": counts.tolist()}


def draw(image, mask, alpha):
    """(overlay, mask picture) of the tracked object: tinted and outlined on the frame, flat on black."""
    overlay, picture = image.copy(), np.zeros_like(image)
    picture[mask] = COLOR
    overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(COLOR) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, COLOR, 2, cv2.LINE_AA)
    return overlay, picture


def points_from(values):
    return [tuple(float(v) for v in point.split(",")) for point in values or []]


def process(model, frames, args, destination, fps):
    h, w = frames.shape[1:3]
    mask = None
    if args.mask:
        mask = cv2.resize(cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_NEAREST) > 127
    prompt = {"points": points_from(args.points), "negative": points_from(args.negative), "mask": mask}
    masks, seconds = track(model, torch.from_numpy(frames).cuda(), args.frame, [prompt], args.direction, args.stride, args.batch)
    masks = masks[0].cpu().numpy()
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "masks.npz", masks=masks)

    def render(frame, index):
        m = masks[index]
        ys, xs = np.nonzero(m)
        objects = [{"id": 1, "box_xyxy": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                    "area": int(m.sum()), "rle": encode_rle(m)}] if len(xs) else []
        overlay, picture = draw(frame.image, m, args.alpha)
        return {"frame": index, "timestamp_seconds": frame.timestamp, "objects": objects}, {"overlay.mp4": overlay, "masks.mp4": picture}

    written = write_run(destination, (Frame(i, i / fps, f) for i, f in enumerate(frames)), render, fps, "masks.jsonl")
    # [Agent] The run's speed is tracking over the frames it covered (tracked or filled), so strides compare fairly.
    first, last = covered_range(args.direction, args.frame, len(masks))
    covered = last - first + 1
    return {"frames": covered, "seconds": seconds, "ms_per_frame": seconds * 1000 / covered, "fps": covered / seconds,
            "write": written}


def run(args):
    if not args.checkpoint.is_file():
        raise SystemExit(f"No checkpoint at {args.checkpoint}; see README › Installation")
    if args.is_image or args.device != "cuda":
        raise SystemExit("sam31 tracks videos on CUDA only")
    if args.mask and args.negative:
        raise SystemExit("--negative refines --points; it cannot be combined with --mask")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = load_tracker(args.checkpoint, args.compile)
    load_seconds = time.perf_counter() - started
    frames = read_video(args.input, args.frames, args.size)
    fps = get_fps(args.input)
    runs = measure(args, lambda destination: process(model, frames, args, destination, fps))
    write_summary(args, {"checkpoint": str(args.checkpoint), "dtype": str(DTYPE), "compile": args.compile,
                         "batch": args.batch, "stride": args.stride, "direction": args.direction,
                         "prompt_frame": args.frame, "points": args.points, "negative": args.negative,
                         "mask": str(args.mask) if args.mask else None, "size_wh": [frames.shape[2], frames.shape[1]],
                         "load_seconds": load_seconds}, runs, fps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    prompt = p.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--points", nargs="+", metavar="X,Y", help="Positive clicks in pixels on --frame")
    prompt.add_argument("--mask", type=Path, help="Mask PNG for --frame (white = object), instead of clicks")
    p.add_argument("--negative", nargs="+", metavar="X,Y", help="Negative clicks in pixels on --frame, with --points")
    p.add_argument("--frame", type=int, default=0, help="Frame the prompt is on; tracking runs from there (default 0)")
    p.add_argument("--direction", choices=["both", "forward", "backward"], default="both")
    p.add_argument("--stride", type=int, default=2, help="Track every Nth frame, fill the rest from neighbours (default 2)")
    p.add_argument("--batch", type=int, default=8, help="Frames per encoder pass (default 8)")
    p.add_argument("--compile", action="store_true", help="torch.compile the image encoder (~2 min once, ~8%% faster)")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--alpha", type=float, default=0.5, help="Mask opacity in the overlay (default 0.5)")
    run(parse(p, "sam31"))


if __name__ == "__main__":
    main()
