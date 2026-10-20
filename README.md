# cv-tools

Local image/video inference. Each model has its own folder and CLI. Sapiens2 is the first model. NVIDIA/Linux selects TensorRT FP16; macOS selects Apple MPS FP16. The tool prints the selected platform, device, precision, and backend before inference.

## Workstation

```bash
cd /home/zuppif/Documents/cv-tools
source .venv/bin/activate

cv-sapiens2 --input inputs/seedance-shopping-reference-6s.mp4
cv-sapiens2 --input photo.jpg --resolution full
```

`cv-tools sapiens2 --input ...` and `python -m cv_tools.sapiens2 --input ...` run the same CLI.

The first TensorRT run exports and builds an engine for that resolution. Later runs reuse it. **You do not need to specify an engine.** The CLI prints the chosen cache path. Use `--engine /path/to/model.plan` only to choose a custom engine location; a missing file is built there. This startup cost is printed separately from warmed inference. Engines are GPU/runtime-specific; build them on the target workstation.

## Resolution

`--resolution` controls **model input**, not the saved image/video dimensions. Output landmarks are mapped back onto the original image.

| Value | Input height × width |
|---|---|
| `full`, `native`, `1` | 1024 × 768; default |
| `half`, `1/2`, `512` | 512 × 384 |
| `quarter`, `1/4`, `256` | 256 × 192 |
| `768` | 768 × 576 |
| `640x480` | Explicit height × width |

A single number sets height and preserves the model's 4:3 height/width ratio, rounding width to the nearest 32. Both dimensions must be at least 32 and divisible by 32. The model's patch size is 16; the CLI deliberately uses the stricter 32-pixel alignment. No separate resizing of the original video happens: the supplied person box is aspect-corrected, padded, and sampled into this model input size.

**Full resolution (1024 × 768) is the default and recommended setting.** Lower input sizes are optional overrides and reduce fine-landmark quality. No ground-truth accuracy assessment has been performed.

## Models

### Sapiens2 · human pose

Runs at **full 1024 × 768 model resolution by default**, preserving the original output-video dimensions. On an idle RTX 3090, two warmed runs measured **17.21 output FPS** with every-frame inference, **26.64 FPS** with 15 FPS inference + linear interpolation, and **31.75 FPS** with 10 FPS inference + linear interpolation. All modes retain every output frame; process VRAM was about **1.67 GiB**. Inspect the overlay for the intended footage.

```bash
cv-sapiens2 --input hd-video.mp4
```

Model code: [`cv_tools/sapiens2/`](cv_tools/sapiens2/). Measured RTX 3090 timings: [`BENCHMARK.md`](cv_tools/sapiens2/BENCHMARK.md).

## Inputs and outputs

```bash
# Fixed person box in original-image pixels: x y width height.
cv-sapiens2 --input photo.jpg --box 100 80 500 900 --output runs/photo

# Short benchmark: five warmup batches, first 30 frames, two measured runs.
cv-sapiens2 --input inputs/seedance-shopping-reference-6s.mp4 \
  --benchmark --output runs/full-benchmark

# All frames, explicit model input dimensions.
cv-sapiens2 --input clip.mp4 --resolution 640x480 --output runs/clip
```

Without `--box`, the entire frame is the person box. There is no detector or tracker. A supplied box stays fixed throughout a video. This is a single-person pose tool; scores are heatmap peaks, not calibrated probabilities.

Each invocation creates a fresh run directory, defaulting to `runs/sapiens2/<timestamp>/`:

```text
summary.json          Environment, timing, GPU memory, landmark names
first-frame.json      First prediction, for comparing backends
run-1/
  landmarks.jsonl     Frame index, timestamp, original-space xy coordinates, scores
  overlay.mp4         Source video with pose drawn on top
  guide.mp4           Skeleton on black
run-2/                Second measured run; --benchmark only
```

Image inputs produce `overlay.jpg` and `guide.png` instead of videos. `--landmarks` selects what is drawn and exported; `--threshold 0.3` controls drawing and eligibility of interpolation endpoints. Raw inferred coordinates/scores are retained for the selected groups. Non-finite predictions are written as JSON `null`. Video outputs are silent MPEG-4 Part 2 (`mp4v`) at the source FPS; no output frames are intentionally skipped.

### Landmark groups

```bash
cv-sapiens2 --input clip.mp4 --landmarks all
cv-sapiens2 --input clip.mp4 --landmarks face
cv-sapiens2 --input clip.mp4 --landmarks body hands
```

| Group | Included landmarks |
|---|---|
| `all` | All 308; default |
| `body` | 22 coarse body/pose anchors, including nose, eyes, ears, and neck |
| `face` | 243 detailed face/ear/eye landmarks and coarse facial anchors |
| `hands` | 42 finger/wrist landmarks |
| `feet` | 8 toe/heel/ankle landmarks |

Groups can be combined and overlapping IDs are deduplicated. JSONL includes original `keypoint_ids`; `summary.json` provides the corresponding names. Filtering does not reduce model-forward cost: the model still predicts all 308 internally.

### Inference sampling and interpolation

Full resolution and every-frame inference remain the defaults. To reduce model evaluations while preserving every output-video frame:

```bash
# Recommended optional compromise: full-resolution inference at 15 FPS, output at source FPS.
cv-sapiens2 --input clip.mp4 --inference-fps 15 --interpolate linear

# Real-time throughput on the tested 30 FPS clip: 31.75 output FPS.
cv-sapiens2 --input clip.mp4 --inference-fps 10

# Hold the previous pose instead of interpolating.
cv-sapiens2 --input clip.mp4 --inference-fps 15 --interpolate none
```

- Sampling uses decoded timestamps, including fractional FPS, rather than assuming every Nth frame. First and final frames are inferred; the final anchor may add one evaluation.
- `linear` is the default when sampling is enabled. It interpolates coordinates on CPU using NumPy and the weaker endpoint score. Low-confidence, non-finite, or off-image endpoints are suppressed rather than bridged.
- Every source frame is decoded, drawn, and exported. Output dimensions and nominal FPS are preserved. Variable-frame-rate input is normalized to constant nominal FPS in the saved MP4; source and output timestamps are both retained in JSONL.
- Interpolation buffers future frames until the next inferred anchor, adding roughly one sample interval of lookahead (about 67 ms at 15 FPS, 100 ms at 10 FPS), plus inference time. This is an offline/file pipeline, not zero-latency live tracking.
- One ordered CPU worker overlaps drawing/encoding with inference; its queue holds at most two outstanding frames. Rendering failures propagate to the CLI. Sampled runs do not preload the whole video into GPU memory.
- The temporal stage works with TensorRT, PyTorch, MPS, CPU, resolution overrides, and landmark groups. Image inputs still receive exactly one prediction. Existing backend/device/batch restrictions apply.
- JSONL marks each frame as `inferred`, `interpolated`, or `held`, with `anchor_frames`; `summary.json` records the actual counts. Interpolated scores are not fresh model observations.

Idle-GPU tests on the six-second reference clip verified **31.75 output FPS at 10 inference FPS**, versus **26.64 at 15 inference FPS** and **17.21 with every-frame inference**. Choose 10 FPS + linear for the tested real-time-throughput mode; 15 FPS + linear has lower interpolation error (**3.48 px median vs 5.61 px**, relative to every-frame predictions). Fast hands remained unreliable in both sampled modes. Keep every-frame inference for fast hands, brief expressions, cuts, or precision-critical motion. These are short-clip throughput results, not a zero-latency live-stream guarantee. Full methodology and both run times are in the model's [BENCHMARK.md](cv_tools/sapiens2/BENCHMARK.md).

## Backends

```bash
cv-sapiens2 --input clip.mp4 --backend tensorrt
cv-sapiens2 --input clip.mp4 --backend compile
cv-sapiens2 --input clip.mp4 --backend torch --resolution full
cv-sapiens2 --input clip.mp4 --backend hf --resolution full
```

- `tensorrt`: cached static-batch-1 FP16 engine; official HF preprocessing and DARK refinement on GPU.
- `compile`: PyTorch FP16 + SDPA + `torch.compile(reduce-overhead)`; compilation is part of startup/warmup.
- `torch`: eager PyTorch FP16 + SDPA; preprocessing and DARK refinement stay on GPU.
- `hf`: baseline matching the original script: CPU preprocessing, FP16 GPU forward, full heatmaps copied to CPU for DARK refinement.
- `channels-last`: retained for reproducing the measured regression; not recommended.

`--batch-size 2` is available for PyTorch throughput experiments. TensorRT currently uses batch 1.

## Device selection

```bash
# Automatic: Linux/NVIDIA -> CUDA/TensorRT/FP16; macOS -> Apple MPS/HF/FP16.
cv-sapiens2 --input photo.jpg

# Explicit device.
cv-sapiens2 --input photo.jpg --device mps
cv-sapiens2 --input clip.mp4 --device cuda
cv-sapiens2 --input photo.jpg --device cpu
```

MPS on Linux fails with an explicit macOS-only error. CUDA on macOS fails with an explicit MPS suggestion. Missing/unavailable GPU support fails rather than silently using CPU. CPU must be explicitly requested and uses HF/FP32. TensorRT and CUDA-specific optimizations fail on MPS/CPU if explicitly selected; MPS uses the same Transformers model, FP16 forward pass, and CPU pre/postprocessing.

Build an engine separately, if desired:

```bash
python -m cv_tools.sapiens2.tensorrt
```

Use `--model PATH` for another local HF-compatible checkpoint and `--engine PATH` for an explicit engine. Use a distinct engine path for each checkpoint and resolution.

## Installation

Linux: Python 3.12 and an NVIDIA GPU/driver supporting CUDA 12.8. macOS: Python 3.12+ and MPS-capable Apple hardware. The setup script installs TensorRT only on Linux.

```bash
bash setup.sh
source .venv/bin/activate
```

Place the existing checkpoint's `config.json`, `preprocessor_config.json`, and `model.safetensors` in `.models/sapiens2-pose-0.4b/`, or supply `--model PATH`. No automatic checkpoint download is performed. `setup.sh` creates this project's own environment; it does not use another application's environment or implementation.

## Measurements

See [the model's BENCHMARK.md](cv_tools/sapiens2/BENCHMARK.md) for measured RTX 3090 speeds. Forward-only timing uses synchronized GPU work with input already resident on the device. Pipeline timing includes decoding, preprocessing, transfer, forward, official DARK refinement, coordinate transfer, drawing, JSONL, both video encoders, and encoder close. Loading, engine building, and warmup are excluded. FPS is total frames divided by measured time, not the encoded playback frame rate.

## Layout

```text
cv_tools/
  common.py           Shared input/output and resolution arguments
  sapiens2/
    cli.py            Image/video inference and two-run benchmark
    tensorrt.py       First-party ONNX export, engine builder, CUDA runtime
    temporal.py       Timestamp sampling, confidence-gated interpolation, frame provenance
    keypoints.json    Meta landmark names, colors, and skeleton links
    BENCHMARK.md      Measured 3090 timings and accuracy/parity caveats
```

Model implementation/checkpoint: [Meta Sapiens2](https://github.com/facebookresearch/sapiens2), [HF checkpoint](https://huggingface.co/facebook/sapiens2-pose-0.4b). Metadata is derived from Meta's pose dataset configuration and remains subject to its license. The inference, export, CLI, and benchmark code here is our implementation; it does not import community inference repositories. Community TensorRT work was used only as a research reference.
