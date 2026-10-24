import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray


def get_fps(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if np.isfinite(fps) and fps > 0 else 30.0


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


def write_run(destination, items, render, fps, records):
    destination.mkdir(parents=True, exist_ok=True)
    writers = {}
    count = 0

    def save(item, index, stream):
        record, files = render(item, index)
        stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        # [Agent] The file name picks the writer: .mp4 appends a video frame, .npy keeps raw floats, anything
        # [Agent] else is a still image. Each tool decides image vs video outputs, so this stays model-agnostic.
        for name, image in files.items():
            if name.endswith(".mp4"):
                if name not in writers:
                    h, w = image.shape[:2]
                    writers[name] = cv2.VideoWriter(str(destination / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writers[name].isOpened():
                        raise OSError(f"Video encoder did not open: {name}")
                writers[name].write(image)
            elif name.endswith(".npy"):
                np.save(destination / name, image)
            elif not cv2.imwrite(str(destination / name), image):
                raise OSError(f"Image output failed: {name}")

    start = time.perf_counter()
    try:
        with (destination / records).open("w") as stream, ThreadPoolExecutor(max_workers=1) as worker:
            pending = deque()
            for item in items:
                # [Agent] One ordered worker draws and encodes while the GPU predicts the next frame. Two
                # [Agent] outstanding frames bound memory, and .result() re-raises write failures here.
                pending.append(worker.submit(save, item, count, stream))
                if len(pending) >= 2:
                    pending.popleft().result()
                count += 1
            for future in pending:
                future.result()
    finally:
        for writer in writers.values():
            writer.release()
    seconds = time.perf_counter() - start
    if count == 0:
        raise ValueError("Input contains no readable frames")
    return {"frames": count, "seconds": seconds, "ms_per_frame": seconds * 1000 / count, "fps": count / seconds}
