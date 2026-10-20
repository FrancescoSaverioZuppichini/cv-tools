import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForPoseEstimation
from transformers.models.sapiens2.image_processing_sapiens2 import (
    box_xywh_to_cxcywh,
    boxes_to_crop_params,
)
from transformers.models.sapiens2.modeling_sapiens2 import Sapiens2PoseEstimatorOutput

from cv_tools.common import (
    IMAGE_SUFFIXES,
    ROOT,
    add_input_arguments,
    resolve_device,
    resolve_input_arguments,
)
from cv_tools.sapiens2.temporal import batches, predictions, read_frames

LANDMARK_GROUPS = {
    "all": list(range(308)),
    "body": list(range(15)) + list(range(63, 70)),
    "face": list(range(5)) + list(range(70, 308)),
    "hands": list(range(21, 63)),
    "feet": [13, 14] + list(range(15, 21)),
}


def arguments():
    p = argparse.ArgumentParser(description="Sapiens2 image/video inference and reproducible short benchmarks")
    add_input_arguments(p)
    p.add_argument("--model", type=Path, default=ROOT / ".models" / "sapiens2-pose-0.4b")
    p.add_argument("--metadata", type=Path, default=Path(__file__).with_name("keypoints.json"))
    p.add_argument("--box", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    p.add_argument("--backend", choices=["auto", "hf", "torch", "channels-last", "compile", "tensorrt"], default="auto")
    p.add_argument("--engine", type=Path)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--landmarks", choices=list(LANDMARK_GROUPS), nargs="+", default=["all"],
                   help="Landmark groups to draw/export; combine with spaces (default: all)")
    p.add_argument("--inference-fps", type=float,
                   help="Pose inference rate; all source frames are still exported (default: every frame)")
    p.add_argument("--interpolate", choices=["linear", "none"], default="linear",
                   help="Fill skipped predictions with linear interpolation or previous pose (default: linear)")
    a = p.parse_args()
    resolve_input_arguments(a, "sapiens2", p)
    resolve_device(a, p)
    if a.backend == "auto":
        a.backend = "tensorrt" if a.device == "cuda" else "hf"
    if a.batch_size < 1 or a.warmup < 1 or a.frames < 0:
        p.error("batch-size/warmup must be positive; frames must be nonnegative")
    if a.box and (not all(np.isfinite(a.box)) or min(a.box[2:]) <= 0):
        p.error("box expects finite x y width height with positive dimensions")
    if a.inference_fps is not None and (not np.isfinite(a.inference_fps) or a.inference_fps <= 0):
        p.error("inference-fps must be a finite positive number")
    if a.device != "cuda" and a.backend != "hf":
        p.error(f"Backend {a.backend} requires CUDA. Use --backend hf for {a.device}; GPU inference still uses FP16.")
    if a.backend == "tensorrt" and a.batch_size != 1:
        p.error("tensorrt currently supports --batch-size 1")
    if a.backend == "tensorrt" and a.engine is None:
        a.engine = ROOT / "engines" / f"sapiens2-fp16-b1-{a.size[0]}x{a.size[1]}.plan"
    return a


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def frames(path, limit):
    for frame in read_frames(path, limit):
        yield frame.image


class Pose:
    def __init__(self, args):
        self.args = args
        self.dtype = torch.float32 if args.device == "cpu" else torch.float16
        self.processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
        self.processor.size = {"height": args.size[0], "width": args.size[1]}
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        self.channels_last = args.backend == "channels-last"
        if args.backend == "tensorrt":
            from cv_tools.sapiens2.tensorrt import TensorRTPose
            self.forward = TensorRTPose(args.engine)
        else:
            self.model = AutoModelForPoseEstimation.from_pretrained(
                args.model, local_files_only=True, dtype=self.dtype, attn_implementation="sdpa"
            ).to(args.device).eval()
            if self.channels_last:
                self.model.to(memory_format=torch.channels_last)
            self.forward = self.model
            if args.backend == "compile":
                self.forward = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
        metadata = json.loads(args.metadata.read_text())
        # [Agent] Numeric IDs define the model's channel order; JSON insertion order is not a contract.
        self.points = [metadata["points"][k] for k in sorted(metadata["points"], key=int)]
        if len(self.points) != config.num_labels:
            raise ValueError("Metadata does not match the model's keypoint count")
        self.names = [p["name"] for p in self.points]
        self.selected = sorted({i for group in args.landmarks for i in LANDMARK_GROUPS[group]})
        self.selected_mask = np.zeros(len(self.points), dtype=bool)
        self.selected_mask[self.selected] = True
        lookup = {name: index for index, name in enumerate(self.names)}
        self.edges = [(lookup[e["link"][0]], lookup[e["link"][1]], tuple(reversed(e["color"])))
                      for e in metadata["skeleton"].values() if all(n in lookup for n in e["link"])]
        self.crop_cache = {}
        self.mean = torch.tensor(self.processor.image_mean, device=args.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_std, device=args.device).view(1, 3, 1, 1)

    def crop(self, image, box):
        h, w = image.shape[:2]
        if min(h, w) < 2:
            raise ValueError("Input image dimensions must be at least 2 pixels")
        key = (h, w, *box)
        if key not in self.crop_cache:
            oh, ow = self.args.size
            source_box = box_xywh_to_cxcywh(torch.tensor([box], dtype=torch.float32, device=self.args.device))
            center, scale = boxes_to_crop_params(source_box, output_size=(oh, ow))
            sx, sy = (ow - 1) / scale[:, 0], (oh - 1) / scale[:, 1]
            gy, gx = torch.meshgrid(torch.arange(oh, dtype=torch.float32, device=self.args.device),
                                    torch.arange(ow, dtype=torch.float32, device=self.args.device), indexing="ij")
            ix = gx / sx[:, None, None] + center[:, 0, None, None] - .5 * scale[:, 0, None, None]
            iy = gy / sy[:, None, None] + center[:, 1, None, None] - .5 * scale[:, 1, None, None]
            grid = torch.stack([2 * ix / (w - 1) - 1, 2 * iy / (h - 1) - 1], dim=-1)
            mode = "bilinear" if min(sx.item(), sy.item()) < 1 else "bicubic"
            self.crop_cache[key] = (grid, mode)
        # [Agent] Video geometry is normally constant. Reuse HF's exact crop grid instead of rebuilding
        # [Agent] it and synchronizing GPU mask decisions for every frame.
        grid, mode = self.crop_cache[key]
        pixels = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.args.device).float()
        crop = torch.nn.functional.grid_sample(pixels, grid, mode=mode, padding_mode="zeros", align_corners=True)
        return ((crop * self.processor.rescale_factor - self.mean) / self.std).to(self.dtype)

    def prepare(self, images):
        boxes = [[self.args.box or [0, 0, image.shape[1], image.shape[0]]] for image in images]
        rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        if self.args.backend == "hf":
            inputs = self.processor([Image.fromarray(x) for x in rgb], boxes=boxes, return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.args.device, dtype=self.dtype)
        else:
            pixels = torch.cat([self.crop(image, box[0]) for image, box in zip(rgb, boxes)])
        if self.channels_last:
            pixels = pixels.contiguous(memory_format=torch.channels_last)
        return pixels, boxes

    def predict(self, pixels, boxes):
        outputs = self.forward(pixel_values=pixels)
        heatmaps = outputs.heatmaps
        if self.args.backend == "hf":
            heatmaps = heatmaps.float().cpu()
        # [Agent] Preserve HF's DARK sub-pixel refinement; only the 308 coordinates/scores leave the GPU.
        results = self.processor.post_process_pose_estimation(
            Sapiens2PoseEstimatorOutput(heatmaps=heatmaps), boxes=boxes
        )
        return [(r[0]["keypoints"].cpu().numpy(), r[0]["scores"].cpu().numpy()) for r in results]

    def draw(self, frame, points, scores):
        guide = np.zeros_like(frame)
        overlay = frame.copy()
        h, w = frame.shape[:2]
        visible = (scores >= self.args.threshold) & np.isfinite(points).all(axis=1)
        visible &= self.selected_mask
        visible &= (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
        xy = np.rint(np.nan_to_num(points)).clip(-100000, 100000).astype(np.int32)
        for a, b, color in self.edges:
            if visible[a] and visible[b]:
                for canvas in (guide, overlay):
                    cv2.line(canvas, tuple(xy[a]), tuple(xy[b]), color, 2, cv2.LINE_AA)
        for i in np.flatnonzero(visible):
            color = tuple(reversed(self.points[i]["color"]))
            for canvas in (guide, overlay):
                cv2.circle(canvas, tuple(xy[i]), 2, color, -1, cv2.LINE_AA)
        return guide, overlay


def write_run(pose, args, destination, fps):
    destination.mkdir(parents=True, exist_ok=True)
    writers = []
    count = 0
    first = None
    kinds = Counter()
    is_image = args.input.suffix.lower() in IMAGE_SUFFIXES

    def save(prediction, index, stream):
        nonlocal writers
        frame, points, scores = prediction.frame.image, prediction.points, prediction.scores
        guide, overlay = pose.draw(frame, points, scores)
        # [Agent] JSON null marks invalid predictions; never emit nonstandard NaN tokens.
        clean_points = [[float(v) if np.isfinite(v) else None for v in xy] for xy in points[pose.selected]]
        clean_scores = [float(v) if np.isfinite(v) else None for v in scores[pose.selected]]
        stream.write(json.dumps({"frame": index, "timestamp_seconds": prediction.frame.timestamp,
                                 "output_timestamp_seconds": index / fps,
                                 "prediction_type": prediction.kind, "anchor_frames": prediction.anchors,
                                 "box_xywh": prediction.box, "keypoints": clean_points,
                                 "keypoint_ids": pose.selected, "scores": clean_scores},
                                separators=(",", ":"), allow_nan=False) + "\n")
        if is_image:
            if not cv2.imwrite(str(destination / "guide.png"), guide) or not cv2.imwrite(str(destination / "overlay.jpg"), overlay):
                raise OSError("Image output failed")
        else:
            if not writers:
                h, w = frame.shape[:2]
                writers = [cv2.VideoWriter(str(destination / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                           for name in ["guide.mp4", "overlay.mp4"]]
                if not all(writer.isOpened() for writer in writers):
                    raise OSError("Video encoder did not open")
            for writer, canvas in zip(writers, (guide, overlay)):
                writer.write(canvas)

    synchronize(args.device)
    start = time.perf_counter()
    try:
        with (destination / "landmarks.jsonl").open("w") as stream, ThreadPoolExecutor(max_workers=1) as output:
            pending = deque()
            for prediction in predictions(pose, args, fps):
                if first is None:
                    first = {"points": prediction.points.tolist(), "scores": prediction.scores.tolist()}
                if is_image:
                    save(prediction, count, stream)
                else:
                    # [Agent] One ordered output worker overlaps CPU drawing/encoding with the next GPU
                    # [Agent] prediction. Two outstanding frames bound memory and propagate write failures.
                    pending.append(output.submit(save, prediction, count, stream))
                    if len(pending) >= 2:
                        pending.popleft().result()
                kinds[prediction.kind] += 1
                count += 1
            for future in pending:
                future.result()
    finally:
        for writer in writers:
            writer.release()
    synchronize(args.device)
    seconds = time.perf_counter() - start
    if count == 0:
        raise ValueError("Input contains no readable frames")
    return {"frames": count, "seconds": seconds, "ms_per_frame": seconds * 1000 / count,
            "fps": count / seconds, "inferred_frames": kinds["inferred"],
            "interpolated_frames": kinds["interpolated"], "held_frames": kinds["held"],
            "actual_inference_fps": None if is_image else kinds["inferred"] * fps / count}, first


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    print(json.dumps({"status": "device", "platform": platform.system(), "device": args.device,
                      "dtype": "float32" if args.device == "cpu" else "float16", "backend": args.backend,
                      "resolution_hw": args.size, "inference_fps": args.inference_fps or "source",
                      "interpolation": args.interpolate if args.inference_fps else "off"}), flush=True)
    if args.backend == "tensorrt":
        print(f"TensorRT engine: {args.engine}", flush=True)
    build_seconds = 0.0
    if args.backend == "tensorrt" and not args.engine.exists():
        from cv_tools.sapiens2.tensorrt import build_engine
        print(f"Building a TensorRT engine for {args.size[0]}x{args.size[1]}; subsequent runs reuse it.", flush=True)
        started = time.perf_counter()
        build_engine(args.model, args.size, args.engine)
        build_seconds = time.perf_counter() - started
        torch.cuda.empty_cache()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = args.backend != "hf"
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pose = Pose(args)
    synchronize(args.device)
    load_seconds = time.perf_counter() - started
    first_group = next(batches(frames(args.input, args.frames), args.batch_size), None)
    if first_group is None:
        raise ValueError("Input contains no readable frames")
    pixels, boxes = pose.prepare(first_group)
    started = time.perf_counter()
    for _ in range(args.warmup):
        pose.predict(pixels, boxes)
    synchronize(args.device)
    warmup_seconds = time.perf_counter() - started
    print(json.dumps({"status": "warm", "load_seconds": load_seconds, "warmup_seconds": warmup_seconds}), flush=True)
    cap = cv2.VideoCapture(str(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    cap.release()
    summary = {"model": "facebook/sapiens2-pose-0.4b",
               "device": torch.cuda.get_device_name() if args.device == "cuda" else args.device,
               "dtype": str(pose.dtype), "size_hw": args.size, "batch_size": args.batch_size,
               "input": str(args.input), "backend": args.backend,
               "platform": platform.system(), "device_type": args.device,
               "torch": torch.__version__, "transformers": transformers.__version__,
               "python": platform.python_version(), "cpu_threads": torch.get_num_threads(),
               "load_seconds": load_seconds, "engine_build_seconds": build_seconds,
               "warmup_seconds": warmup_seconds, "warmup_batches": args.warmup,
               "landmark_names": [pose.names[i] for i in pose.selected], "landmark_ids": pose.selected,
               "landmark_groups": args.landmarks,
               "requested_inference_fps": args.inference_fps,
               "interpolation": args.interpolate if args.inference_fps else "off",
               "output_worker": "One ordered CPU worker; at most two outstanding frames; worker drain and encoder close included",
               "timestamp_policy": "Sample decoded timestamps; missing/repeated times use nominal FPS. MP4 output is constant nominal FPS; JSON preserves source/output times separately.",
               "score_semantics": "Measured scores are heatmap peaks; interpolated scores are the weaker endpoint, or zero when suppressed. Not calibrated probabilities.",
               "box_policy": "User-supplied fixed xywh, otherwise full frame; no person detector or tracking",
               "forward_scope": ("Not measured separately for sampled runs; pipeline timing only" if args.inference_fps
                                 else "Synchronized forward only, all input frames preprocessed on GPU before timing"),
               "pipeline_scope": "Decode, RGB, official crop/normalize, transfer, forward, official DARK refinement, transfer coordinates, draw, JSONL, two video encoders and close; warmup/loading excluded"}
    if args.benchmark and args.inference_fps is None:
        cached = [pose.prepare(group)[0] for group in batches(frames(args.input, args.frames), args.batch_size)]
        forward_ms = []
        for _ in range(2):
            synchronize(args.device)
            start = time.perf_counter()
            for pixels in cached:
                result = pose.forward(pixel_values=pixels)
            synchronize(args.device)
            forward_ms.append((time.perf_counter() - start) * 1000 / sum(x.shape[0] for x in cached))
            del result
        summary["forward_run_ms_per_frame"] = forward_ms
        summary["forward_mean_ms_per_frame"] = statistics.mean(forward_ms)
        del cached
    runs = []
    for run in range(2 if args.benchmark else 1):
        stats, first = write_run(pose, args, args.out / f"run-{run + 1}", fps)
        runs.append(stats)
        if run == 0:
            (args.out / "first-frame.json").write_text(json.dumps(first))
        print(json.dumps({"run": run + 1, **stats}), flush=True)
    summary["runs"] = runs
    summary["mean_ms_per_frame"] = statistics.mean(r["ms_per_frame"] for r in runs)
    summary["throughput_fps"] = 1000 / summary["mean_ms_per_frame"]
    summary["source_fps"] = fps
    summary["real_time"] = summary["throughput_fps"] >= fps
    if args.device == "cuda":
        summary["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        summary["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        summary["process_vram_mib_at_end"] = None
        try:
            memory = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True
            )
            summary["process_vram_mib_at_end"] = next(
                (int(row.split(",")[1]) for row in memory.splitlines() if int(row.split(",")[0]) == os.getpid()), None
            )
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in {"landmark_names", "landmark_ids"}}, indent=2), flush=True)


def main():
    args = arguments()
    if args.device == "cuda":
        # [Agent] TensorRT adds device-wide synchronization on the default stream. Keep preprocessing,
        # [Agent] inference and refinement on one explicit stream so their ordering stays correct without it.
        with torch.cuda.stream(torch.cuda.Stream()):
            run(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
