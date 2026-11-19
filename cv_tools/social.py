"""Social: an image through the path a real phone photo takes before anyone sees it: phone processing, Instagram, a WhatsApp re-send"""
import argparse
import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cv_tools import ROOT

SHARPEN = 2.0


def compress(image, quality):
    """(decoded image, JPEG bytes). 4:2:0 like phones, Instagram and WhatsApp: colour at half resolution is part of the look."""
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality, subsampling=2, optimize=True)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"), buffer.getvalue()


def develop(image):
    # [Agent] The phone's own processing: chroma noise reduction only, then hard ISP sharpening. Settings from a sweep against
    # real social photos: luma denoise left them waxy, and the 1080 px downscale plus JPEG eat so much texture that only
    # strong sharpening (2.0) keeps them near the real photos' numbers.
    bgr = cv2.fastNlMeansDenoisingColored(np.asarray(image)[:, :, ::-1].copy(), None, 0.01, 3, 7, 21)
    bgr = cv2.addWeighted(bgr, 1 + SHARPEN, cv2.GaussianBlur(bgr, (0, 0), 1.2), -SHARPEN, 0)
    return Image.fromarray(bgr[:, :, ::-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, nargs="+", required=True, help="Images")
    p.add_argument("--output", type=Path, help="Output folder; default: runs/social/<timestamp>")
    # [Agent] PNG-noised images are already the phone's processed sRGB (SIDD noise was captured after the iPhone ISP):
    # denoising and sharpening them again would erase the real noise.
    p.add_argument("--no-phone", action="store_true", help="Skip the phone stage, for inputs that already carry real phone noise (png_noise, phone_finish)")
    args = p.parse_args()
    out = args.output or ROOT / "runs" / "social" / datetime.now().strftime("%Y%m%d-%H%M%S")
    for path in args.input:
        image = Image.open(path).convert("RGB")
        # [Agent] Two folder levels in the output path: renders from different scenes share leaf names.
        destination = out / path.parent.parent.name / path.parent.name
        destination.mkdir(parents=True, exist_ok=True)
        phone, data = compress(image if args.no_phone else develop(image), 90)
        (destination / f"{path.stem}-phone.jpg").write_bytes(data)
        # [Agent] Instagram serves 1080 px wide; area resampling is what an upload's downscale does to fine detail.
        if phone.width > 1080:
            phone = phone.resize((1080, round(phone.height * 1080 / phone.width)), Image.Resampling.BOX)
        instagram, data = compress(phone, 78)
        (destination / f"{path.stem}-ig.jpg").write_bytes(data)
        (destination / f"{path.stem}-wa.jpg").write_bytes(compress(instagram, 70)[1])
        print(destination / path.stem)


if __name__ == "__main__":
    main()
