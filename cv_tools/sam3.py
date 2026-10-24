"""SAM 3: text-prompted instance masks, tracked across video frames"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from cv_tools import MODEL_DIR
from cv_tools.cli import measure, parse, write_summary
from cv_tools.media import get_fps, read_frames, write_run


def track(model, processor, frames, args):
    # [Agent] Streaming session: frames go in one at a time and only the tracker memory stays on the GPU,
    # [Agent] so VRAM does not grow with clip length. An image is a one-frame video through the same path.
    session = processor.init_video_session(inference_device=args.device, video_storage_device="cpu", dtype=model.dtype)
    processor.add_text_prompt(session, args.prompt)
    for frame in frames:
        inputs = processor(images=cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB), device=args.device, return_tensors="pt")
        outputs = model(inference_session=session, frame=inputs.pixel_values[0])
        result = processor.postprocess_outputs(session, outputs, original_sizes=inputs.original_sizes)
        prompts = {i: text for text, ids in result["prompt_to_obj_ids"].items() for i in ids}
        yield frame, [{"id": i, "prompt": prompts[i], "score": float(score), "box": box.tolist(), "mask": mask}
                      for i, score, box, mask in zip(result["object_ids"].tolist(), result["scores"].float(),
                                                     result["boxes"].float().cpu(), result["masks"].cpu().numpy())]


def encode_rle(mask):
    # [Agent] COCO uncompressed RLE: run lengths over the column-major mask, starting with a background run,
    # [Agent] so pycocotools decodes it losslessly. The MP4 masks are lossy and only for viewing.
    flat = mask.ravel(order="F")
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate([[0], edges, [flat.size]]))
    if flat[0]:
        counts = np.concatenate([[0], counts])
    return {"size": list(mask.shape), "counts": counts.tolist()}


def get_color(object_id):
    # [Agent] Golden-ratio hue steps keep neighbouring IDs far apart, and an ID keeps its colour across frames.
    hue = int((object_id * 0.618034 % 1) * 180)
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def draw(image, objects, alpha):
    overlay = image.copy()
    masks = np.zeros_like(image)
    for obj in objects:
        color, mask = get_color(obj["id"]), obj["mask"]
        masks[mask] = color
        overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)
        x, y = int(obj["box"][0]), max(int(obj["box"][1]) - 6, 12)
        cv2.putText(overlay, f"{obj['prompt']} #{obj['id']} {obj['score']:.2f}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return overlay, masks


def process(model, processor, args, destination, fps):
    def render(item, index):
        frame, objects = item
        overlay, masks = draw(frame.image, objects, args.alpha)
        record = {"frame": index, "timestamp_seconds": frame.timestamp, "output_timestamp_seconds": index / fps,
                  "objects": [{"id": o["id"], "prompt": o["prompt"], "score": o["score"], "box_xyxy": o["box"],
                               "area": int(o["mask"].sum()), "rle": encode_rle(o["mask"])} for o in objects]}
        if args.is_image:
            return record, {"overlay.jpg": overlay, "masks.png": masks}
        return record, {"overlay.mp4": overlay, "masks.mp4": masks}

    frames = read_frames(args.input, args.frames, fps)
    return write_run(destination, track(model, processor, frames, args), render, fps, "masks.jsonl")


@torch.inference_mode()
def run(args):
    if not (args.model / "config.json").is_file():
        raise SystemExit(f"No checkpoint at {args.model}; see README › Installation")
    if not all(0 <= v <= 1 for v in (args.threshold, args.new_threshold, args.alpha)):
        raise SystemExit("thresholds and --alpha must be between 0 and 1")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    print(f"sam3: {args.device}, prompts {args.prompt}", flush=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = Sam3VideoModel.from_pretrained(args.model, local_files_only=True, dtype=dtype).to(args.device).eval()
    model.score_threshold_detection = args.threshold
    model.new_det_thresh = args.new_threshold
    processor = Sam3VideoProcessor.from_pretrained(args.model, local_files_only=True)
    load_seconds = time.perf_counter() - started
    first = list(read_frames(args.input, 1))
    started = time.perf_counter()
    for _ in range(args.warmup):
        list(track(model, processor, first, args))
    warmup_seconds = time.perf_counter() - started
    fps = get_fps(args.input)
    runs = measure(args, lambda destination: process(model, processor, args, destination, fps))
    summary = {"model": str(args.model), "dtype": str(dtype), "prompts": args.prompt, "threshold": args.threshold,
               "new_threshold": args.new_threshold, "load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
               "warmup_passes": args.warmup}
    write_summary(args, summary, runs, fps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", nargs="+", required=True, help='Concepts to segment, e.g. --prompt person "red bag"')
    p.add_argument("--model", type=Path, default=MODEL_DIR / "sam3")
    p.add_argument("--threshold", type=float, default=0.5, help="Detection score kept on each frame (default: 0.5)")
    p.add_argument("--new-threshold", type=float, default=0.7, help="Detection score that starts a new object (default: 0.7)")
    p.add_argument("--alpha", type=float, default=0.5, help="Mask opacity in overlay (default: 0.5)")
    run(parse(p, "sam3"))


if __name__ == "__main__":
    main()
