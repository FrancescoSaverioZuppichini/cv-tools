"""Phone finish: sharpen, then real iPhone 7 sensor noise (PNG), saved as a high-quality JPEG at the same size"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from cv_tools.media import IMAGE_SUFFIXES
from cv_tools.png_noise import PROFILES, get_prompt, load_model, sample_noise


def finish(model, image, prompt, args, scale):
    # [Agent] Noise is sampled for the clean picture, then added on top of the sharpened one: sharpening after the noise
    # would sharpen the grain too (the crunchier option, voted below this one).
    # [Agent] scale > 1: the grain is sampled on the picture shrunk by scale and enlarged back, and the sharpen radius grows
    # with it, so a 2x upscale gets the same visible finish as the render it came from.
    h, w = image.shape[:2]
    small = image if scale == 1 else cv2.resize(image, (round(w / scale), round(h / scale)), interpolation=cv2.INTER_AREA)
    noise = sample_noise(model, small, prompt, args.seed, args.batch) * args.blend
    if scale != 1:
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    sharp = image + args.sharpen * (image - cv2.GaussianBlur(image, (0, 0), 1.2 * scale))
    return np.clip(sharp + noise, 0, 1)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, nargs="+", required=True, help="Images and/or folders (folders: every image directly inside)")
    p.add_argument("--output", type=Path, help="Output folder; default: <input folder>-finished next to it")
    p.add_argument("--sharpen", type=float, default=1.0, help="Unsharp amount at radius 1.2 px; 0 = off (default: 1.0)")
    p.add_argument("--profile", choices=list(PROFILES), default="D", help="iPhone 7 noise reference: A=ISO 1000 … D=2000 (default: D)")
    p.add_argument("--blend", type=float, help="Noise strength 0-1; default: the profile's subtle blend (D = 0.28)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95, full-resolution colour)")
    p.add_argument("--max-side", type=int, help="Shrink (Lanczos) so the long side is at most this before finishing")
    p.add_argument("--look-scale", type=float, help="Finish as if the picture were this many times smaller (grain size and sharpen "
                                                    "radius grow with it); default: long side / 1536, at least 1")
    p.add_argument("--seed", type=int, default=2601801)
    p.add_argument("--batch", type=int, default=16, help="256 px tiles per model call (default: 16)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    files = [(file, source) for source in args.input
             for file in (sorted(f for f in source.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES) if source.is_dir() else [source])]
    if not files:
        p.error("no images found")
    args.blend = PROFILES[args.profile][2] if args.blend is None else args.blend
    model = load_model(args.device)
    prompt = get_prompt(model, args.profile)
    for file, source in files:
        started = time.perf_counter()
        out = args.output or (source.with_name(source.name + "-finished") if source.is_dir() else file.parent.with_name(file.parent.name + "-finished"))
        out.mkdir(parents=True, exist_ok=True)
        image = Image.open(file).convert("RGB")
        if args.max_side and max(image.size) > args.max_side:
            k = args.max_side / max(image.size)
            image = image.resize((round(image.width * k), round(image.height * k)), Image.Resampling.LANCZOS)
        scale = args.look_scale or max(1.0, max(image.size) / 1536)
        finished = finish(model, np.asarray(image, np.float32) / 255, prompt, args, scale)
        # [Agent] subsampling=0 keeps colour at full resolution (4:4:4); the grain's colour speckle survives the JPEG.
        Image.fromarray(np.round(finished * 255).astype(np.uint8)).save(out / f"{file.stem}.jpg", quality=args.quality, subsampling=0, optimize=True)
        print(f"{file.name} → {out / file.stem}.jpg · {image.width}×{image.height} · look scale {scale:g} · {time.perf_counter() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
