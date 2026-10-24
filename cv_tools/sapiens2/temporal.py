from dataclasses import dataclass

import numpy as np

from cv_tools.media import Frame, read_frames


@dataclass
class Prediction:
    frame: Frame
    points: np.ndarray
    scores: np.ndarray
    box: list
    kind: str
    anchors: tuple[int, ...]


def infer(pose, frame):
    points, scores, box = pose.predict(frame.image)
    return Prediction(frame, points, scores, box, "inferred", (frame.index,))


def between(left, right, frame, threshold):
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
    frames = read_frames(args.input, args.frames, fps)
    if args.inference_fps is None or args.is_image:
        yield from (infer(pose, frame) for frame in frames)
        return
    gaps, left, deadline = [], None, 0.0
    frame = next(frames, None)
    while frame is not None:
        following = next(frames, None)
        # [Agent] Sample on decoded timestamps (fractional and variable FPS included), and always infer the
        # [Agent] final frame so the tail is never frozen. One frame of lookahead tells us which one is last.
        if frame.timestamp + 1e-7 >= deadline or following is None:
            deadline = (np.floor((frame.timestamp + 1e-7) * args.inference_fps) + 1) / args.inference_fps
            right = infer(pose, frame)
            yield from (between(left, right, gap, args.threshold) for gap in gaps)
            yield right
            gaps, left = [], right
        else:
            gaps.append(frame)
        frame = following
