"""Sapiens2 body-part contours: per-part outlines on black, one colour per part"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor
from transformers.models.sapiens2.modeling_sapiens2 import Sapiens2ForSemanticSegmentation

from cv_tools import MODEL_DIR

# [Agent] The config ships only LABEL_n and the Goliath (Sapiens v1) order does not match Sapiens2 (checked against pose keypoints: 10 of 34 limb labels landed on their own limb), so parts stay unnamed: each label gets its own fixed colour and the map shows shapes, not left / right claims.
PALETTE = [tuple(int(c) for c in cv2.cvtColor(np.uint8([[[i * 179 // 29, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]) for i in range(29)]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Image or folder of images")
    p.add_argument("--output", type=Path, required=True, help="Folder for <name>.png contours and <name>-labels.png label maps")
    p.add_argument("--model", type=Path, default=MODEL_DIR / "sapiens2-seg-0.4b")
    p.add_argument("--min-area", type=float, default=0.0005, help="Drop part fragments smaller than this share of the image")
    args = p.parse_args()
    files = [args.input] if args.input.is_file() else sorted(f for f in args.input.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = Sapiens2ForSemanticSegmentation.from_pretrained(args.model, local_files_only=True, dtype=torch.float16).cuda().eval()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for f in files:
        image = cv2.imread(str(f))
        h, w = image.shape[:2]
        inputs = processor(images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), return_tensors="pt").to("cuda", torch.float16)
        labels = processor.post_process_semantic_segmentation(model(**inputs), target_sizes=[(h, w)])[0].cpu().numpy().astype(np.uint8)
        cv2.imwrite(str(args.output / f"{f.stem}-labels.png"), labels)
        canvas = np.zeros_like(image)
        found = {}
        for i in np.unique(labels):
            if i == 0:
                continue
            mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [c for c in contours if cv2.contourArea(c) >= args.min_area * h * w]
            name = f"label_{i}"
            cv2.drawContours(canvas, contours, -1, PALETTE[i % len(PALETTE)], 2, cv2.LINE_AA)
            ys, xs = np.nonzero(mask)
            found[name] = {"label": int(i), "pixels": int(mask.sum()), "centroid_xy": [round(float(xs.mean()) / w, 3), round(float(ys.mean()) / h, 3)]}
        cv2.imwrite(str(args.output / f"{f.stem}.png"), canvas)
        summary[f.name] = found
        print(f.name, "parts", len(found), flush=True)
    (args.output / "parts.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
