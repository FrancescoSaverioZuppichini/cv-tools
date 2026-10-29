# cv-tools

Local image/video inference. Every tool is one Python file you run directly; the model tools share the same flags and output layout.

| Run | What | Output |
|---|---|---|
| `python -m cv_tools.sapiens2.pose` | [Meta Sapiens2 0.4B](https://github.com/facebookresearch/sapiens2) pose | 308 landmarks, skeleton overlay |
| `python -m cv_tools.depth_anything` | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) small/base/large | relative depth map |
| `python -m cv_tools.sam3` | [Meta SAM 3](https://huggingface.co/facebook/sam3) | text-prompted instance masks, tracked across video frames |
| `python -m cv_tools.palette` | k-means in Lab | dominant colours, highlight and shadow tint, printed to the terminal |
| `python -m cv_tools.canny` | OpenCV Canny, auto thresholds | edge map (white on black), image or video |

Every file takes `--help`.

## Workstation

```bash
cd /home/zuppif/Documents/cv-tools
source .venv/bin/activate

python -m cv_tools.sapiens2.pose --input inputs/seedance-shopping-reference-6s.mp4
python -m cv_tools.depth_anything --input photo.jpg
python -m cv_tools.sam3 --input clip.mp4 --prompt person "shopping bag"
python -m cv_tools.palette --input photo.jpg
python -m cv_tools.canny --input photo.jpg
```

## Model tools: shared flags and outputs

| Flag | Meaning |
|---|---|
| `--input PATH` | Image (`jpg png webp bmp tif`) or video; required |
| `--output DIR` | Fresh run directory; default `runs/<tool>/<timestamp>/`; a non-empty directory is refused |
| `--frames N` | Only the first N frames; default all, or 30 with `--benchmark` |
| `--benchmark` | Two complete measured runs instead of one |
| `--warmup N` | Warmup passes on the first frame before timing; default 5 |
| `--device auto\|cuda\|mps\|cpu` | `auto` is CUDA on Linux, MPS on macOS. A missing GPU fails instead of silently using CPU |

Every run writes:

```text
summary.json          Environment, settings, load/warmup time, per-run timing, FPS, GPU memory
run-1/
  <records>.jsonl     One line per frame: index, source and output timestamps, the model's result
  <images or videos>  Per tool, below
run-2/                --benchmark only
```

Images give still files; videos give silent `mp4v` MP4s at the source FPS with every frame kept. Drawing and encoding run on one background thread while the GPU predicts the next frame. Timing covers decode, inference, drawing, JSONL and encoding; loading, engine builds and warmup are excluded.

## Sapiens2 · human pose

```bash
python -m cv_tools.sapiens2.pose --input clip.mp4                     # full 1024×768, every frame
python -m cv_tools.sapiens2.pose --input clip.mp4 --inference-fps 10  # infer at 10 FPS, interpolate the rest
python -m cv_tools.sapiens2.pose --input photo.jpg --box 100 80 500 900 --landmarks body hands
```

| Flag | Meaning |
|---|---|
| `--resolution full\|half\|quarter\|HxW` | Model input, not output size: 1024×768 (default), 512×384, 256×192, or sides divisible by 32 |
| `--box X Y W H` | Fixed person box in source pixels; default the whole frame. No detector or tracker, single person |
| `--landmarks all\|body\|face\|hands\|feet` | Groups to draw/export, combinable; the model still predicts all 308 |
| `--threshold 0.3` | Score to draw a point and to trust it as an interpolation endpoint |
| `--inference-fps N` | Infer at N FPS, linearly interpolate the frames between |
| `--model PATH` | Default `.models/sapiens2-pose-0.4b` |

Outputs `landmarks.jsonl` (original-space xy, heatmap-peak scores, `keypoint_ids`, `null` for invalid points), `overlay.(jpg|mp4)` (pose on the source) and `guide.(png|mp4)` (skeleton on black).

**Backend.** CUDA runs a static FP16 TensorRT engine with FP32 normalization guards; MPS runs PyTorch FP16, CPU PyTorch FP32. Preprocessing is HF's crop and normalization as one GPU `grid_sample`, and HF's DARK sub-pixel refinement runs on the GPU too. The first CUDA run per resolution builds `engines/<model>-<HxW>.plan` (minutes, reported as `engine_build_seconds`); later runs load it in about a second. Engines are GPU/driver specific: build on the target machine.

**Sampling.** `--inference-fps` samples on decoded timestamps (fractional and variable FPS included), always infers the first and final frames, and fills the rest by linear interpolation. Interpolated points take the weaker endpoint score; low-confidence, non-finite or off-image endpoints are dropped rather than bridged. It looks ahead to the next inferred frame, so it is an offline pipeline, not live tracking. JSONL marks each frame `inferred` or `interpolated` with its `anchor_frames`.

**Speed**, idle RTX 3090, 720×1280 clip, full resolution: **17.21 FPS** every frame, **26.64 FPS** at `--inference-fps 15`, **31.75 FPS** at 10. About 1.7 GiB VRAM. 15 FPS stays closer to every-frame predictions (3.48 px median vs 5.61 px at 10); keep every frame for fast hands, cuts or precision. Each refactor was checked against the recorded 10 FPS run: identical landmarks, 0.0 px. Methodology and history: [`BENCHMARK.md`](cv_tools/sapiens2/BENCHMARK.md).

## Depth Anything V2 · relative depth

```bash
python -m cv_tools.depth_anything --input photo.jpg                       # large, 518 short side
python -m cv_tools.depth_anything --input clip.mp4 --model base --colormap turbo
```

| Flag | Meaning |
|---|---|
| `--model small\|base\|large\|PATH` | Checkpoint size, or a local HF checkpoint; default `large` |
| `--resolution N` | Model input short side, multiple of 14; default 518, the training size. Long side keeps the aspect ratio |
| `--colormap inferno\|turbo\|magma\|viridis\|gray` | Preview colours; default `inferno` |

| Image | Video | Content |
|---|---|---|
| `depth.png` | `depth.mp4` | Depth, near = white: 16-bit PNG, 8-bit video |
| `color.jpg` | `color.mp4` | Colormapped preview |
| `depth.npy` | — | Raw float32 model output at source resolution |
| `depth.jsonl` | `depth.jsonl` | Per frame `min`/`max` of the raw output |

The model predicts **relative inverse depth**: bigger is nearer, no metric scale. Each frame is stretched to its own min/max, saved in `depth.jsonl` so files map back: `raw = min + value / 65535 · (max − min)`, `/ 255` for video. Per-frame stretching can flicker slightly on video. Resize and normalization run on the GPU and match the HF processor (Pearson 0.99998).

**Speed**, idle RTX 3090, FP16, 720×1280 input (518×924 model input), forward per frame: **small 12.8 ms, base 26 ms, large 61 ms**. The full 180-frame clip with large: **15.3 FPS**, GPU-bound, 1.7 GiB VRAM. Plain PyTorch with SDPA attention is the fastest option measured: `torch.compile` was ~2× slower, and an FP16 TensorRT engine with FP32 norm guards ~15% slower (interleaved test on a shared GPU). Small is Apache-2.0; base and large are CC-BY-NC-4.0.

## SAM 3 · text-prompted segmentation

```bash
python -m cv_tools.sam3 --input photo.jpg --prompt person
python -m cv_tools.sam3 --input clip.mp4 --prompt person "red bag" --threshold 0.4
```

| Flag | Meaning |
|---|---|
| `--prompt TEXT...` | One or more noun phrases; every instance of each is found; required |
| `--threshold 0.5` | Detection score kept on each frame (model default) |
| `--new-threshold 0.7` | Detection score that starts a new tracked object (model default) |
| `--alpha 0.5` | Mask opacity in the overlay |
| `--model PATH` | Default `.models/sam3` |

Outputs `overlay.(jpg|mp4)` (masks, outlines and `prompt #id score`), `masks.(png|mp4)` (each object in its own colour on black) and `masks.jsonl`: per frame, each object's `id`, `prompt`, `score`, `box_xyxy`, `area` and a lossless COCO uncompressed `rle` mask. Decode with `from pycocotools import mask; mask.decode(mask.frPyObjects(rle, *rle["size"]))`. Use the RLE, not the lossy MP4, for downstream masks.

Images and videos run through one `Sam3VideoModel` streaming session: frames go in one at a time, object IDs stay stable across a video, and memory does not grow with clip length. Streaming skips SAM 3's hotstart filter, which needs future frames, so an object seen for only a few frames is kept rather than retroactively removed. BF16 on GPU, FP32 on CPU. **Not yet run end-to-end: the gated checkpoint is not downloaded.**

## Palette · colours and light

```bash
python -m cv_tools.palette --input photo.jpg                   # table with terminal swatches
python -m cv_tools.palette --input photo.jpg --format hex      # one #RRGGBB per line
python -m cv_tools.palette --input clip.mp4 --format json --colors 8
```

```text
           #767166  rgb(118, 113, 102)  L  47.9   25.4%
           #0E0B0A  rgb( 14,  11,  10)  L   3.2   23.3%
           ...
highlight  #C1A092  rgb(193, 160, 146)  L  68.6
shadow     #060203  rgb(  6,   2,   3)  L   0.9
mean lightness 36.0 / 100
```

| Flag | Meaning |
|---|---|
| `--format text\|hex\|rgb\|json` | `text` (default) is the table above; `hex` and `rgb` print one palette colour per line; `json` has everything |
| `--colors 6` | Palette size, 1–32 |
| `--sample-fps 1` | Video frames sampled per second; all samples share one palette |

Colours are k-means clusters in CIE Lab, where distance follows perceived difference, sorted by pixel share. `L` is Lab lightness 0–100. `highlight` and `shadow` are the mean colour of the brightest and darkest 10 % of pixels: the key-light tint and the shadow/ambient tint to prompt with. Frames are downscaled to ~64k pixels first; an image takes ~0.3 s including Python start-up. CPU only, no torch.

## Installation

Linux: Python 3.12 and an NVIDIA driver supporting CUDA 12.8. macOS: Python 3.12+ and MPS-capable hardware. TensorRT is installed only on Linux.

```bash
bash setup.sh
source .venv/bin/activate
```

Checkpoints live in `.models/` and load offline; nothing downloads at run time. `CV_TOOLS_DATA=/path` moves `.models/` and `engines/` to that folder (the workstation keeps them on `/media/zuppif/Cancro/cv-tools`); the download commands below then take `$CV_TOOLS_DATA/.models/...`.

```bash
hf download facebook/sapiens2-pose-0.4b --local-dir .models/sapiens2-pose-0.4b
hf download depth-anything/Depth-Anything-V2-Small-hf --local-dir .models/depth-anything-v2-small
hf download depth-anything/Depth-Anything-V2-Base-hf --local-dir .models/depth-anything-v2-base
hf download depth-anything/Depth-Anything-V2-Large-hf --local-dir .models/depth-anything-v2-large
# SAM 3 is gated: accept the license at huggingface.co/facebook/sam3, run `hf auth login`, then
hf download facebook/sam3 --local-dir .models/sam3 --exclude sam3.pt
```

`sam3.pt` is the 3.4 GB original-format duplicate of `model.safetensors`; the tool does not need it.

## Layout

```text
cv_tools/
  cli.py              Shared flags, device selection, summary.json
  media.py            Frame reading with timestamps; ordered output worker writing JSONL, images, videos
  tensorrt.py         ONNX export to a static FP16 engine with FP32 norm guards; engine runtime
  depth_anything.py   GPU preprocessing, depth, colormaps
  sam3.py             Streaming segmentation/tracking, overlay, RLE masks
  palette.py          Lab k-means palette, highlight/shadow tint
  sapiens2/
    pose.py           GPU crop, TensorRT/PyTorch forward, DARK refinement, drawing
    temporal.py       Timestamp sampling, confidence-gated interpolation
    keypoints.json    Meta landmark names, colours, skeleton links
    BENCHMARK.md      RTX 3090 measurements
```

A new model tool is one file: a `main()` that adds its own flags to an `ArgumentParser`, calls `cli.parse(parser, name)` for the shared ones, and passes the result to its `run(args)`.

```bash
python -m unittest discover -s tests
```

The inference, export and CLI code is our own on top of Transformers' model implementations; it does not import community inference repositories. Sapiens2 landmark metadata derives from Meta's pose dataset configuration and keeps its license.

## Canny · edge map

```bash
python -m cv_tools.canny --input photo.jpg                    # auto thresholds from the median brightness
python -m cv_tools.canny --input clip.mp4 --low 50 --high 150 # fixed thresholds
```

| Flag | Meaning |
|---|---|
| `--low N` / `--high N` | Hysteresis thresholds; default auto: (1 ± sigma) × median brightness |
| `--sigma 0.33` | Spread of the auto thresholds around the median |
| `--blur 5` | Odd Gaussian kernel before edge detection, so skin and fabric texture do not become speckle; `0` disables |
| `--frames N` | Frame limit for videos; default all |

Outputs `canny.png` (image) or `canny.mp4` (video, source FPS) and `edges.jsonl` (per frame: the thresholds used and the edge pixel count). CPU only, no model download.

