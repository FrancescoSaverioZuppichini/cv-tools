"""Find the shots (hard cuts) in a video and write shots.json plus one frame strip per shot, so whoever writes the prompt can
look at each shot and use the cut times. One shot means no timestamps in an H3 prompt; several mean [Shot N] At MM:SS.mmm.

    python -m cv_tools.shots --input clip.mp4 --output runs/shots/clip
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

# [Agent] Fixed before measuring on our clips (2026-11-22): the highest scene score on 13 single-shot phone clips was 0.138, a
# real cut scored 0.526. A new angle inside the same room scores lower and is untested, so check the strips when in doubt.
CUT = 0.3
MIN_SHOT = 12
EVERY = 6


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL).stdout


def scene_scores(path):
    out = run(["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf", "select='gte(scene,0)',metadata=print:file=-", "-f", "null", "-"])
    return [float(x) for x in re.findall(r"scene_score=([\d.]+)", out)]


def shot_starts(score, cut, min_shot):
    # [Agent] A cut must leave at least min_shot frames on both sides, so a flash or a fast pan right at the edges never splits a shot.
    starts = [0]
    for k, s in enumerate(score):
        if s >= cut and k - starts[-1] >= min_shot and len(score) - k >= min_shot:
            starts.append(k)
    return starts


def stamp(t):
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


def strip(path, a, b, out):
    # [Agent] Frames picked by index, never by time seek, so the strip shows exactly the frames the model follows.
    idx = sorted(set(range(a, b, EVERY)) | {b - 1})
    cols = min(7, len(idx))
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-an", "-vf", f"select={sel},scale=480:-2,tile={cols}x{-(-len(idx) // cols)}",
         "-frames:v", "1", "-vsync", "0", str(out)])
    return idx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Directory for shots.json and shot-N.png")
    p.add_argument("--cut", type=float, default=CUT, help=f"Scene score that starts a new shot (default {CUT})")
    p.add_argument("--min-shot", type=int, default=MIN_SHOT, help=f"Shortest shot in frames (default {MIN_SHOT})")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    num, den = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(a.input)]).strip().split("/")
    fps = float(num) / float(den)
    score = scene_scores(a.input)
    starts = shot_starts(score, a.cut, a.min_shot)
    ends = starts[1:] + [len(score)]
    shots = []
    for n, (s, e) in enumerate(zip(starts, ends), 1):
        png = a.output / f"shot-{n}.png"
        shots.append({"shot": n, "start_frame": s, "end_frame": e - 1, "start": stamp(s / fps), "end": stamp(e / fps),
                      "cut_score": round(score[s], 3) if n > 1 else None, "strip": str(png), "strip_frames": strip(a.input, s, e, png)})
    result = {"input": str(a.input), "fps": fps, "frames": len(score), "shots": shots,
              "max_score_inside_shots": round(max((x for k, x in enumerate(score) if k not in starts), default=0.0), 3)}
    (a.output / "shots.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
