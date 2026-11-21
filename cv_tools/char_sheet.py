"""Cut a three-column character sheet into its parts with Sapiens2 pose: the front and back bodies from just under the
chin to just under the feet, and the head framed on the face. Crops only, at the input's full resolution.

The sheet has three columns split by thin white gaps: full body from the front, full body from the back, and a column
whose top half is a face close-up. Both bodies get the same two cuts, so they keep the same scale; the head box is
centred between the eyes.

    python -m cv_tools.char_sheet --input sheet.png
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw

from cv_tools.cli import parse
from cv_tools.motion_guide import load_pose, predict_person
from cv_tools.sapiens2.pose import pose_model


def runs(mask):
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def columns(sheet):
    """The three columns' x ranges and the row where the third column's top panel ends."""
    grey = np.asarray(sheet.convert("L"), float)
    h, w = grey.shape
    # [Agent] Gaps are whiter than the studio grey: the cut-off sits halfway between the background (the left edge
    # strip, always floor and wall) and pure white, and a gap is a run of columns that clear it from top to bottom.
    white = (float(np.median(grey[:, : max(4, w // 200)])) + 255) / 2
    gaps = [r for r in runs(np.percentile(grey, 5, axis=0) > white) if 0 < r[0] and r[1] < w - 1]
    if len(gaps) < 2:
        raise SystemExit(f"expected two white gaps between three columns, found {len(gaps)}")
    cols = [(0, gaps[0][0]), (gaps[0][1] + 1, gaps[1][0]), (gaps[1][1] + 1, w)]
    rows = [r for r in runs(np.percentile(grey[:, cols[2][0]:], 5, axis=1) > white) if 0 < r[0] and r[1] < h - 1]
    # [Agent] The divider under the face close-up is often too faint to find; the panels are then halves.
    return cols, rows[0][0] if rows else h // 2


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pose-model", default="0.8b", help="Sapiens2 pose size (0.4b | 0.8b | 1b | 5b) or a checkpoint path (default 0.8b)")
    p.add_argument("--score", type=float, default=0.3, help="keypoint confidence to trust (default 0.3)")
    p.add_argument("--below-chin", type=float, default=0.1, help="body top cut under the chin, share of chin-to-shoulders (default 0.1)")
    p.add_argument("--floor", type=float, default=0.03, help="floor kept under the lowest foot point, share of the body height (default 0.03)")
    p.add_argument("--hair", type=float, default=0.75, help="head top above the brows, share of brows-to-chin (default 0.75)")
    p.add_argument("--under-chin", type=float, default=0.12, help="head bottom under the chin, share of brows-to-chin (default 0.12)")
    p.add_argument("--aspect", type=float, default=0.8, help="head width over height (default 0.8)")
    p.add_argument("--grow", type=float, default=1.12, help="head box scale around its centre (default 1.12)")
    a = parse(p, "char_sheet")
    if not a.is_image:
        p.error("--input must be a picture of the sheet")
    a.out.mkdir(parents=True, exist_ok=True)
    sheet = Image.open(a.input).convert("RGB")
    w, h = sheet.size
    cols, split = columns(sheet)

    pose = load_pose(a.score, "all", "full", pose_model(a.pose_model))
    idx = {n: i for i, n in enumerate(pose.names)}
    chin_i, eyes = idx["tip_of_chin"], [idx["left_eye"], idx["right_eye"]]
    shoulders_i = [idx["left_shoulder"], idx["right_shoulder"]]
    feet_i = [idx[f"{s}_{part}"] for s in ("left", "right") for part in ("ankle", "big_toe", "small_toe", "heel")]

    def find(box):
        # [Agent] Sapiens reads BGR; only points it is sure of count.
        pts, sc = predict_person(pose, np.ascontiguousarray(np.asarray(sheet.crop(box))[:, :, ::-1]))
        return pts, lambda i: sc[i] >= a.score and np.isfinite(pts[i]).all()

    views = [find((x0, 0, x1, h)) for x0, x1 in cols[:2]]
    (fp, fok), (bp, bok) = views
    if not (fok(chin_i) and all(fok(i) for i in shoulders_i)):
        raise SystemExit("no chin or shoulders on the front view")
    chin, shoulders = float(fp[chin_i][1]), float(np.mean([fp[i][1] for i in shoulders_i]))
    top = round(chin + a.below_chin * (shoulders - chin))
    feet = [p_[i][1] for p_, ok in views for i in feet_i if ok(i)]
    if not feet:
        raise SystemExit("no foot points found")
    lowest = float(max(feet))
    bottom = min(h, round(lowest + a.floor * (lowest - shoulders)))
    back = [float(bp[i][1]) for i in shoulders_i if bok(i)]
    if back and top >= min(back):
        raise SystemExit(f"the top cut (y {top}) would clip the back view's shoulders (y {min(back):.0f}); lower --below-chin")

    # [Agent] The head box is framed on the face landmarks and centred between the eyes, then kept symmetric inside the
    # panel, so the face is always in the middle (a hair outline is not used: pale hair on grey escapes it).
    px0, pw = cols[2][0], w - cols[2][0]
    hp, hok = find((px0, 0, w, split))
    face = np.array([hp[i] for i in pose.openpose_face if hok(i)])
    if len(face) < 20 or not hok(chin_i) or not all(hok(i) for i in eyes):
        raise SystemExit("face landmarks not found on the face panel")
    brows, hchin = float(face[:, 1].min()), float(hp[chin_i][1])
    span = hchin - brows
    top0, bottom0 = brows - a.hair * span, hchin + a.under_chin * span
    cy, height = (top0 + bottom0) / 2, (bottom0 - top0) * a.grow
    cx = float(np.mean([hp[i][0] for i in eyes]))
    half = min(a.aspect * height / 2, cx, pw - cx)
    head_box = (px0 + round(cx - half), max(0, round(cy - height / 2)), px0 + round(cx + half), min(split, round(cy + height / 2)))

    sheet.crop((cols[0][0], top, cols[0][1], bottom)).save(a.out / "front.png")
    sheet.crop((cols[1][0], top, cols[1][1], bottom)).save(a.out / "back.png")
    sheet.crop(head_box).save(a.out / "head.png")

    proof = sheet.copy()
    d = ImageDraw.Draw(proof)
    r = max(4, h // 250)
    for (x0, x1), (p_, ok) in zip(cols[:2], views):
        for i in (chin_i, *shoulders_i, *feet_i):
            if ok(i):
                d.ellipse([x0 + p_[i][0] - r, p_[i][1] - r, x0 + p_[i][0] + r, p_[i][1] + r], fill=(0, 200, 0))
        for y in (top, bottom):
            d.line([x0, y, x1, y], fill=(230, 0, 0), width=r)
    for x, y in face[::4]:
        d.ellipse([px0 + x - r, y - r, px0 + x + r, y + r], fill=(0, 200, 0))
    d.rectangle(head_box, outline=(230, 0, 0), width=r)
    proof.thumbnail((2400, 2400))
    proof.save(a.out / "proof.jpg", quality=90)

    summary = {"input": str(a.input), "size": [w, h], "columns": cols, "face_panel_rows": [0, split],
               "body": {"chin": round(chin), "shoulders": round(shoulders), "back_shoulders": [round(v) for v in back],
                        "lowest_foot": round(lowest), "rows": [top, bottom]},
               "head": {"box": head_box, "off_centre_px": round(cx - (head_box[0] + head_box[2]) / 2 + px0, 1)},
               "settings": {k: getattr(a, k) for k in ("pose_model", "score", "below_chin", "floor", "hair", "under_chin", "aspect", "grow")}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"front/back rows {top}..{bottom} ({bottom - top} px tall), head {head_box[2] - head_box[0]}x{head_box[3] - head_box[1]} "
          f"(off centre {summary['head']['off_centre_px']:+} px) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
