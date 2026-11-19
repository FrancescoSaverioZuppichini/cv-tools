"""Sapiens2 pose: 308 body, face, hand and foot landmarks"""
import argparse
import colorsys
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForPoseEstimation
from transformers.models.sapiens2.image_processing_sapiens2 import box_xywh_to_cxcywh, boxes_to_crop_params
from transformers.models.sapiens2.modeling_sapiens2 import Sapiens2PoseEstimatorOutput

from cv_tools import ENGINE_DIR, MODEL_DIR
from cv_tools.cli import measure, parse, write_summary
from cv_tools.media import get_fps, read_frames, write_run
from cv_tools.sapiens2.temporal import predictions
from cv_tools.tensorrt import Engine, create_engine

KEYPOINTS = Path(__file__).with_name("keypoints.json")
LANDMARK_GROUPS = {
    "all": list(range(308)),
    "body": list(range(15)) + list(range(63, 70)),
    "face": list(range(5)) + list(range(70, 308)),
    "hands": list(range(21, 63)),
    "feet": [13, 14] + list(range(15, 21)),
}
RESOLUTIONS = {"full": (1024, 768), "half": (512, 384), "quarter": (256, 192)}

# [Agent] --style dwpose draws the same Sapiens2 points the way DWPose / OpenPose control maps look: OpenPose-18 body with its fixed
# [Agent] per-limb colours (that is how pose ControlNets tell left from right and bone from bone), 21-point rainbow hands, white face dots.
# [Agent] Pose control LoRAs are trained on that picture, not on coordinates, so the Sapiens drawing has to speak their visual language.
OPENPOSE_BODY = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
                 "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]
OPENPOSE_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
OPENPOSE_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
                   (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
HAND_FINGERS = ["thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger"]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20)]
FACE_PARTS = ("eyebrow", "lash_line", "nose", "lip", "nostril", "mouth")


def resolution(value):
    if value in RESOLUTIONS:
        return RESOLUTIONS[value]
    if not re.fullmatch(r"\d+x\d+", value):
        raise argparse.ArgumentTypeError("resolution expects full, half, quarter, or HxW like 640x480")
    size = tuple(map(int, value.split("x")))
    if min(size) < 32 or any(n % 32 for n in size):
        raise argparse.ArgumentTypeError("resolution sides must be at least 32 and divisible by 32")
    return size


class Pose:
    def __init__(self, args):
        self.args = args
        self.dtype = torch.float32 if args.device == "cpu" else torch.float16
        self.processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
        if args.device == "cuda":
            self.forward = Engine(args.engine)
        else:
            model = AutoModelForPoseEstimation.from_pretrained(
                args.model, local_files_only=True, dtype=self.dtype, attn_implementation="sdpa").to(args.device).eval()
            self.forward = lambda pixels: model(pixel_values=pixels).heatmaps  # noqa: E731
        metadata = json.loads(KEYPOINTS.read_text())
        # [Agent] Numeric IDs define the model's channel order; JSON insertion order is not a contract.
        self.points = [metadata["points"][k] for k in sorted(metadata["points"], key=int)]
        if len(self.points) != AutoConfig.from_pretrained(args.model, local_files_only=True).num_labels:
            raise ValueError("keypoints.json does not match the model's keypoint count")
        self.names = [p["name"] for p in self.points]
        self.selected = sorted({i for group in args.landmarks for i in LANDMARK_GROUPS[group]})
        self.selected_mask = np.isin(np.arange(len(self.points)), self.selected)
        lookup = {name: index for index, name in enumerate(self.names)}
        self.edges = [(lookup[e["link"][0]], lookup[e["link"][1]], tuple(reversed(e["color"])))
                      for e in metadata["skeleton"].values() if all(n in lookup for n in e["link"])]
        self.openpose_body = [lookup.get(n) for n in OPENPOSE_BODY]
        self.openpose_hands = [[lookup.get(f"{side}_wrist")] + [lookup.get(f"{side}_{finger}{joint}") for finger in HAND_FINGERS for joint in ("_third_joint", "2", "3", "4")]
                               for side in ("left", "right")]
        # [Agent] Face: brows, eyes, nose and lips only; Sapiens2 has no jaw contour, so this is the "no jaw" face DWPose users strip by hand.
        self.openpose_face = [i for i, n in enumerate(self.names) if any(part in n for part in FACE_PARTS) and not n.startswith(("left_", "right_"))]
        self.mean = torch.tensor(self.processor.image_mean, device=args.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_std, device=args.device).view(1, 3, 1, 1)
        self.grids = {}

    def get_grid(self, h, w, box):
        # [Agent] HF's crop as one grid_sample: the box is aspect-corrected and padded exactly like the processor,
        # [Agent] then every model pixel samples its source point. Built once per frame size, reused every frame.
        oh, ow = self.args.resolution
        device = self.args.device
        center, scale = boxes_to_crop_params(box_xywh_to_cxcywh(torch.tensor([box], dtype=torch.float32, device=device)),
                                             output_size=(oh, ow))
        sx, sy = (ow - 1) / scale[:, 0], (oh - 1) / scale[:, 1]
        gy, gx = torch.meshgrid(torch.arange(oh, dtype=torch.float32, device=device),
                                torch.arange(ow, dtype=torch.float32, device=device), indexing="ij")
        ix = gx / sx[:, None, None] + center[:, 0, None, None] - .5 * scale[:, 0, None, None]
        iy = gy / sy[:, None, None] + center[:, 1, None, None] - .5 * scale[:, 1, None, None]
        grid = torch.stack([2 * ix / (w - 1) - 1, 2 * iy / (h - 1) - 1], dim=-1)
        return grid, "bilinear" if min(sx.item(), sy.item()) < 1 else "bicubic"

    def predict(self, image, box=None):
        """(keypoints, scores, box) of the person in box (x, y, w, h); default: the CLI's --box, else the whole frame."""
        h, w = image.shape[:2]
        if box is None:
            # [Agent] The CLI's box is fixed for the whole run, so its crop grid is built once per frame size.
            box = self.args.box or [0, 0, w, h]
            if (h, w) not in self.grids:
                self.grids[h, w] = self.get_grid(h, w, box)
            grid, mode = self.grids[h, w]
        else:
            grid, mode = self.get_grid(h, w, box)
        # [Agent] as_tensor: a frame already on the GPU is used in place; a NumPy frame is uploaded.
        pixels = torch.as_tensor(image, device=self.args.device).permute(2, 0, 1)[None].flip(1).float()
        crop = torch.nn.functional.grid_sample(pixels, grid, mode=mode, padding_mode="zeros", align_corners=True)
        heatmaps = self.forward(((crop * self.processor.rescale_factor - self.mean) / self.std).to(self.dtype))
        # [Agent] HF's DARK sub-pixel refinement runs where the heatmaps are; only 308 points leave the GPU.
        result = self.processor.post_process_pose_estimation(Sapiens2PoseEstimatorOutput(heatmaps=heatmaps),
                                                             boxes=[[box]])[0][0]
        return result["keypoints"].cpu().numpy(), result["scores"].cpu().numpy(), box

    def draw_dwpose(self, frame, points, scores):
        canvas = np.zeros_like(frame)
        h, w = frame.shape[:2]

        def ok(i):
            return (i is not None and scores[i] >= self.args.threshold and np.isfinite(points[i]).all()
                    and 0 <= points[i][0] < w and 0 <= points[i][1] < h)

        def bgr(rgb, scale=1.0):
            return tuple(int(c * scale) for c in reversed(rgb))

        body = self.openpose_body
        for k, (a, b) in enumerate(OPENPOSE_LIMBS):
            if ok(body[a]) and ok(body[b]):
                (x1, y1), (x2, y2) = points[body[a]], points[body[b]]
                # [Agent] Limbs as filled ellipses at 60 % colour, joints as full-colour dots, as in the OpenPose renderer the control LoRAs saw.
                length, angle = math.hypot(x1 - x2, y1 - y2), math.degrees(math.atan2(y1 - y2, x1 - x2))
                polygon = cv2.ellipse2Poly((int((x1 + x2) / 2), int((y1 + y2) / 2)), (int(length / 2), 4), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(canvas, polygon, bgr(OPENPOSE_COLORS[k], 0.6))
        for k, i in enumerate(body):
            if ok(i):
                cv2.circle(canvas, (int(points[i][0]), int(points[i][1])), 4, bgr(OPENPOSE_COLORS[k]), -1)
        for hand in self.openpose_hands:
            for k, (a, b) in enumerate(HAND_EDGES):
                if ok(hand[a]) and ok(hand[b]):
                    color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(k / len(HAND_EDGES), 1.0, 1.0))
                    cv2.line(canvas, tuple(int(v) for v in points[hand[a]]), tuple(int(v) for v in points[hand[b]]), bgr(color), 2)
            for i in hand:
                if ok(i):
                    cv2.circle(canvas, (int(points[i][0]), int(points[i][1])), 4, bgr((0, 0, 255)), -1)
        for i in self.openpose_face:
            if ok(i):
                cv2.circle(canvas, (int(points[i][0]), int(points[i][1])), 3, (255, 255, 255), -1)
        overlay = frame.copy()
        drawn = canvas.any(axis=2)
        overlay[drawn] = canvas[drawn]
        return canvas, overlay

    def draw(self, frame, points, scores):
        if self.args.style == "dwpose":
            return self.draw_dwpose(frame, points, scores)
        guide = np.zeros_like(frame)
        overlay = frame.copy()
        h, w = frame.shape[:2]
        visible = (scores >= self.args.threshold) & np.isfinite(points).all(axis=1) & self.selected_mask
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


def process(pose, args, destination, fps):
    kinds = Counter()

    def render(prediction, index):
        kinds[prediction.kind] += 1
        guide, overlay = pose.draw(prediction.frame.image, prediction.points, prediction.scores)
        # [Agent] JSON null marks invalid predictions; never emit nonstandard NaN tokens.
        points = [[float(v) if np.isfinite(v) else None for v in xy] for xy in prediction.points[pose.selected]]
        scores = [float(v) if np.isfinite(v) else None for v in prediction.scores[pose.selected]]
        record = {"frame": index, "timestamp_seconds": prediction.frame.timestamp,
                  "output_timestamp_seconds": index / fps, "prediction_type": prediction.kind,
                  "anchor_frames": prediction.anchors, "box_xywh": prediction.box,
                  "keypoints": points, "keypoint_ids": pose.selected, "scores": scores}
        if args.is_image:
            return record, {"guide.png": guide, "overlay.jpg": overlay}
        return record, {"guide.mp4": guide, "overlay.mp4": overlay}

    stats = write_run(destination, predictions(pose, args, fps), render, fps, "landmarks.jsonl")
    return {**stats, "inferred_frames": kinds["inferred"], "interpolated_frames": kinds["interpolated"],
            "actual_inference_fps": None if args.is_image else kinds["inferred"] * fps / stats["frames"]}


def pose_model(name):
    """A size (0.4b, 0.8b, 1b, 5b) means .models/sapiens2-pose-<size>; anything with a slash is a checkpoint path."""
    name = str(name)
    path = Path(name).expanduser() if "/" in name else MODEL_DIR / f"sapiens2-pose-{name.lower()}"
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"no checkpoint at {path}; download it with: "
                                         f"hf download facebook/{path.name} --local-dir {path}")
    return path


def pose_engine(model, resolution):
    """The TensorRT engine for this checkpoint and model input, built on first use (a couple of minutes). Returns (path, build seconds)."""
    import tensorrt

    h, w = resolution
    # [Agent] A plan only loads in the TensorRT version that built it; with the version in the name an upgrade builds a fresh one.
    engine = ENGINE_DIR / f"{model.name}-{h}x{w}-trt{tensorrt.__version__}.plan"
    if engine.exists():
        return engine, 0.0
    started = time.perf_counter()
    create_engine(AutoModelForPoseEstimation.from_pretrained(
        model, local_files_only=True, dtype=torch.float16, attn_implementation="sdpa").cuda().eval(),
        "heatmaps", resolution, engine)
    return engine, time.perf_counter() - started


@torch.inference_mode()
def run(args):
    if args.box and (not all(np.isfinite(args.box)) or min(args.box[2:]) <= 0):
        raise SystemExit("--box expects finite x y width height with positive dimensions")
    if args.inference_fps is not None and (not np.isfinite(args.inference_fps) or args.inference_fps <= 0):
        raise SystemExit("--inference-fps must be a finite positive number")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    h, w = args.resolution
    print(f"sapiens2: {args.device}, {'tensorrt' if args.device == 'cuda' else 'torch'}, {args.model.name}, model input {h}x{w}", flush=True)
    args.engine, build_seconds = pose_engine(args.model, args.resolution) if args.device == "cuda" else (None, 0.0)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = True
    # [Agent] TensorRT adds device-wide synchronization on the default stream. Keep preprocessing, inference
    # [Agent] and refinement on one explicit stream so their ordering stays correct without it.
    if args.device == "cuda":
        torch.cuda.set_stream(torch.cuda.Stream())
    started = time.perf_counter()
    pose = Pose(args)
    load_seconds = time.perf_counter() - started
    first = next(read_frames(args.input, 1, size=args.size)).image
    started = time.perf_counter()
    for _ in range(args.warmup):
        pose.predict(first)
    warmup_seconds = time.perf_counter() - started
    fps = get_fps(args.input)
    runs = measure(args, lambda destination: process(pose, args, destination, fps))
    summary = {"model": str(args.model), "engine": str(args.engine) if args.device == "cuda" else None,
               "dtype": str(pose.dtype), "size_hw": args.resolution, "load_seconds": load_seconds,
               "engine_build_seconds": build_seconds, "warmup_seconds": warmup_seconds, "warmup_passes": args.warmup,
               "landmark_groups": args.landmarks, "landmark_ids": pose.selected,
               "landmark_names": [pose.names[i] for i in pose.selected], "inference_fps": args.inference_fps}
    write_summary(args, summary, runs, fps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=pose_model, default="0.4b",
                   help="Checkpoint size (0.4b | 0.8b | 1b | 5b, from .models/sapiens2-pose-<size>) or a path (default: 0.4b)")
    p.add_argument("--resolution", type=resolution, default=(1024, 768),
                   help="Model input: full | half | quarter | HxW (default: full, 1024x768)")
    p.add_argument("--box", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                   help="Fixed person box in source pixels (default: whole frame)")
    p.add_argument("--threshold", type=float, default=0.3, help="Score to draw a point or trust it for interpolation")
    p.add_argument("--landmarks", choices=list(LANDMARK_GROUPS), nargs="+", default=["all"],
                   help="Landmark groups to draw/export; combine with spaces (default: all)")
    p.add_argument("--style", choices=["sapiens", "dwpose"], default="sapiens",
                   help="Drawing: sapiens (all selected landmarks, Sapiens colours) or dwpose (OpenPose-18 body, hands, face dots, the look pose ControlNets expect)")
    p.add_argument("--inference-fps", type=float,
                   help="Infer at this rate and interpolate the frames between (default: every frame)")
    run(parse(p, "sapiens2"))


if __name__ == "__main__":
    main()
