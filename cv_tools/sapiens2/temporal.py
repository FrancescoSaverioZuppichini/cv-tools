from collections import deque
from dataclasses import dataclass
from itertools import islice

import cv2
import numpy as np
from PIL import Image

from cv_tools.common import IMAGE_SUFFIXES


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray


@dataclass
class Prediction:
    frame: Frame
    points: np.ndarray
    scores: np.ndarray
    box: list
    kind: str
    anchors: tuple[int, ...]


def batches(items, size):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def read_frames(path, limit, fps=30.0):
    if path.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            yield Frame(0, 0.0, cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR))
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    origin = None
    previous = -1.0
    index = 0
    try:
        while not limit or index < limit:
            ok, image = cap.read()
            if not ok:
                break
            raw = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if origin is None:
                origin = raw if np.isfinite(raw) and raw >= 0 else 0.0
            timestamp = raw - origin
            # [Agent] Use decoded presentation times, including fractional/VFR cadence. Some decoders
            # [Agent] report missing or repeated timestamps; keep those streams monotonic at nominal FPS.
            if not np.isfinite(timestamp) or timestamp <= previous:
                timestamp = 0.0 if index == 0 else previous + 1 / fps
            yield Frame(index, timestamp, image)
            previous = timestamp
            index += 1
    finally:
        cap.release()


def infer(pose, frames):
    pixels, boxes = pose.prepare([f.image for f in frames])
    predictions = pose.predict(pixels, boxes)
    return [Prediction(frame, points, scores, box[0], "inferred", (frame.index,))
            for frame, (points, scores), box in zip(frames, predictions, boxes)]


def between(left, right, frame, threshold, method):
    if method == "none":
        return Prediction(frame, left.points, left.scores, left.box, "held", left.anchors)
    weight = (frame.timestamp - left.frame.timestamp) / (right.frame.timestamp - left.frame.timestamp)
    h, w = frame.image.shape[:2]
    valid = np.ones(len(left.scores), dtype=bool)
    for endpoint in (left, right):
        xy = endpoint.points
        valid &= np.isfinite(xy).all(axis=1) & np.isfinite(endpoint.scores) & (endpoint.scores >= threshold)
        valid &= (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    points = np.full_like(left.points, np.nan)
    points[valid] = left.points[valid] + weight * (right.points[valid] - left.points[valid])
    # [Agent] Interpolation is not a model observation. Use the weaker endpoint score and suppress
    # [Agent] invalid/off-frame endpoints so unstable DARK peaks cannot sweep across the image.
    scores = np.where(valid, np.minimum(left.scores, right.scores), 0.0)
    return Prediction(frame, points, scores, left.box, "interpolated", (left.frame.index, right.frame.index))


def predictions(pose, args, fps):
    source = read_frames(args.input, args.frames, fps)
    if args.inference_fps is None or args.input.suffix.lower() in IMAGE_SUFFIXES:
        for group in batches(source, args.batch_size):
            yield from infer(pose, group)
        return
    pending = deque()
    anchors = []
    left = None
    last = None
    deadline = 0.0

    def flush():
        nonlocal left
        for right in infer(pose, anchors):
            while pending and pending[0].index <= right.frame.index:
                frame = pending.popleft()
                if frame.index == right.frame.index:
                    yield right
                else:
                    yield between(left, right, frame, args.threshold, args.interpolate)
            left = right
        anchors.clear()

    for frame in source:
        pending.append(frame)
        last = frame
        if frame.timestamp + 1e-7 >= deadline:
            anchors.append(frame)
            deadline = (np.floor((frame.timestamp + 1e-7) * args.inference_fps) + 1) / args.inference_fps
        if len(anchors) == args.batch_size:
            yield from flush()
    if last is not None:
        last_anchor = anchors[-1].index if anchors else left.frame.index
        # [Agent] Infer the final frame too: no frozen/extrapolated tail, even for truncated clips.
        if last.index != last_anchor:
            anchors.append(last)
        if anchors:
            yield from flush()
