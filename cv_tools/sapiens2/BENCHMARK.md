# Sapiens2 0.4B measurements

RTX 3090 24 GB, PyTorch 2.11.0+cu128, Transformers 5.20.1, TensorRT 10.16.1.11. FP16 with FP32 normalization constraints, batch 1 unless specified. Four PyTorch CPU threads; one OpenCV thread.

## Current pipeline · verified idle-GPU results

Full **1024×768** model input, all 308 landmarks, entire `seedance-shopping-reference-6s.mp4`: **180 frames, 720×1280, 30 FPS**. Five warmup passes, then **two complete measured runs per configuration**. Same cached engine and bounded CPU output worker throughout. No competing GPU compute processes were detected during any of the three invocations; monitoring records are saved with each result.

| Inference sampling | Evaluations / run | Interpolated frames / run | Pipeline run 1 | Pipeline run 2 | Mean ms/output frame | Output FPS | Speedup | At least 30 FPS? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Every frame (30 FPS) | 180 | 0 | 10.4268 s | 10.4894 s | 58.10 | 17.21 | 1.00× | No |
| 15 FPS + linear | 91 | 89 | 6.8237 s | 6.6907 s | 37.54 | 26.64 | 1.55× | No |
| **10 FPS + linear** | **61** | **119** | **5.7137 s** | **5.6240 s** | **31.49** | **31.75** | **1.84×** | **Yes, on this clip** |

Mean clip-processing times: **10.4581 / 6.7572 / 5.6688 seconds** respectively. All configurations retained all 180 output frames and six-second duration; both overlay and skeleton videos were validated for both runs. Process VRAM at completion was **1710 MiB (1.67 GiB)** in each configuration; this is a snapshot, not a peak measurement.

Timing includes decoding, GPU preprocessing, inference, official DARK refinement, coordinate transfer, temporal interpolation, drawing, JSONL, both mp4v encoders, output-worker drain, and encoder close. Cached-engine loading (0.88–1.77 s), warmup (0.41–0.45 s), and later H.264 browser-preview conversion are excluded. No engine builds were needed. GPU monitoring ran outside the timed CLI pipeline.

The every-frame control uses explicit `--inference-fps 30`: all 180 frames were inferred, none interpolated. This keeps the same sampling code path across configurations and avoids the unrelated forward-only microbenchmark in all three tests. The final frame is always inferred, hence 61/91 evaluations rather than 60/90. FPS is total frames divided by total measured pipeline time across the two runs.

Quality checks against the **new every-frame GPU run** reproduce the earlier CPU-replay results exactly: median/p95 deviation **5.61/18.73 px** at 10 FPS and **3.48/14.87 px** at 15 FPS, on mutually visible/confident interpolated points. Inferred anchor coordinates match the every-frame run with **0.0 px maximum absolute difference** in both configurations. This measures agreement with the model, not ground-truth accuracy; fast-hand limitations remain.

**Use 10 FPS + linear for the measured real-time-throughput mode; use 15 FPS + linear for lower interpolation error, or every-frame inference for detail.** The 10 FPS result has about 5.8% throughput headroom on this clip, not a guarantee for every video. Interpolation still adds lookahead and this is not a zero-latency live-stream benchmark. Full resolution/every-frame inference remains the default.

```bash
python -m cv_tools.sapiens2.pose --input inputs/seedance-shopping-reference-6s.mp4 --inference-fps 10

# Reproduce one configuration, five warmup passes and two complete runs.
python -m cv_tools.sapiens2.pose --input inputs/seedance-shopping-reference-6s.mp4 \
  --inference-fps 10 --frames 180 --benchmark
```

Raw runs: `runs/gpu-idle-every-frame/`, `runs/gpu-idle-15-linear/`, and `runs/gpu-idle-10-linear/`. Each includes `summary.json`, `gpu-monitor.json`, both measured runs and browser-playable `preview.mp4`/`skeleton.mp4`. Sampled configurations also include `quality-vs-every-frame.json`.

## Earlier optimization experiments

First 30 frames of the same clip. Five warmup batches, two measured runs. These historical rows precede the final output-worker implementation and are not the current pipeline's speed. The `compile`, `channels-last` and batch-2 backends were removed afterwards; TensorRT beat all of them.

| Configuration | Model input H×W | Forward mean ms/frame | Pipeline run 1 | Pipeline run 2 | Pipeline mean ms/frame | FPS | Speedup vs HF |
|---|---|---:|---:|---:|---:|---:|---:|
| HF GPU baseline | 1024×768 | 103.03 | 11.119 s | 10.696 s | 363.59 | 2.75 | 1.00× |
| GPU preprocessing + DARK | 1024×768 | 103.69 | 3.864 s | 3.895 s | 129.33 | 7.73 | 2.81× |
| GPU + channels-last | 1024×768 | 109.20 | 4.084 s | 4.048 s | 135.52 | 7.38 | 2.68× |
| GPU + compile | 1024×768 | 78.40 | 3.154 s | 3.128 s | 104.69 | 9.55 | 3.47× |
| GPU + compile, batch 2 | 1024×768 | 76.15 | 3.049 s | 3.060 s | 101.83 | 9.82 | 3.57× |
| GPU + compile, half resolution | 512×384 | 18.43 | 1.271 s | 1.275 s | 42.44 | 23.56 | 8.57× |
| TensorRT FP16, half | 512×384 | 11.42 | 1.043 s | 1.054 s | 34.95 | 28.62 | 10.40× |
| TensorRT FP16, half, explicit CUDA stream | 512×384 | 11.35 | 1.048 s | 1.025 s | 34.54 | 28.95 | 10.52× |
| TensorRT, FP32 normalization guard + cached crops | 512×384 | 12.86 | 1.193 s | 1.180 s | 39.56 | 25.28 | 9.19× |

The first six rows were measured before the CLI-only packaging refactor, using the same inference/benchmark implementation and the workstation's pre-existing library environment. TensorRT measurements use the isolated `cv-tools` environment; versions are recorded in each `summary.json`.

Full-clip half-resolution validation: all 180 frames processed in **7.360 s**, or **24.46 FPS**, after warmup. Process VRAM at completion: **1544 MiB**. Peak allocated tensor memory: **0.938 GiB**; peak PyTorch reserved memory: **1.172 GiB**. These are different metrics; the sampled process footprint is not a peak measurement.

Full-clip native-resolution validation with cached crop grids: all 180 frames processed in **17.972 s**, or **10.02 FPS**, after warmup. Input 1024×768, compiled PyTorch FP16. Process VRAM at completion: **1942 MiB**; peak tensor allocation: **1.081 GiB**. This is one full-clip validation run, separate from the two-run/30-frame table.

### Full-resolution TensorRT · one pass, as requested

| GPU | Backend | Model input H×W | Frames | Pipeline time | FPS | Process VRAM |
|---|---|---|---:|---:|---:|---:|
| RTX 3090 24 GB | TensorRT FP16 + FP32 normalization | 1024×768 | 180 | 13.982 s | 12.87 | 1738 MiB |

This is a **single pass over the full six-second video**, not a two-run average. One warmup pass (**0.125 s**), model/engine load (**0.817 s**), and first engine build (**138.392 s**) are excluded. The original full-resolution benchmark invocation was stopped before inference when the request changed to a single visual-result run. The existing ONNX export was reused. H.264 conversion for the separate preview page is also outside the pipeline measurement; the timed outputs use mp4v.

Backend parity over all 180 full-resolution frames: **49,519 mutually confident points** compared against compiled PyTorch; median displacement **0.0697 px**, p95 **0.4023 px**, mean absolute score difference **0.00182**. All **55,440** TensorRT coordinates were finite. Raw maximum displacement was **5634.66 px** because DARK can extrapolate unstable heatmap peaks far outside the image; these points are excluded from drawing by the normal bounds check, but preserved in raw output. This is backend parity, not annotated-dataset accuracy.

Final CLI checks: CUDA/TensorRT image inference defaults to 1024×768 and exports exactly 22 points for `--landmarks body`; macOS image/video inference selects MPS FP16 and combined `face hands` exports 285 unique IDs. MPS on Linux and CUDA on macOS both fail with explicit platform errors. The cached GPU crop differs from the official processor by at most **one FP16 step (0.0009765625)** on the checked frame.

TensorRT version: **10.16.1.11**. Half-resolution ONNX export took **33.85 s**; initial engine build took **83.43 s**. Engine load approximately **0.92 s**, five warmed batches **0.20 s**. Process VRAM after the stream test: **1348 MiB**. PyTorch's allocated-memory counter excludes TensorRT allocations and is not an estimate of total TensorRT VRAM.

The initial weakly typed TensorRT engine needed a precision correction: against compiled PyTorch at the same half resolution, mutually confident landmarks differed by a median **2.37 px**, p95 **12.68 px**, with large outliers. Those first two TensorRT rows are **rejected speed experiments**, not the shipped engine.

Keeping float32 normalization/reduction operations in float32 reduced median displacement to **0.187 px**, p95 to **1.491 px**, and mean absolute score difference to **0.00228**, over the same 30 frames. Raw maximum displacement remains **1492 px**, from unstable/extrapolated DARK coordinates; all 9240 output points were finite. This checks backend parity, not ground-truth accuracy. The precision-corrected engine is deliberately retained despite its slower timing. Process VRAM: **1394 MiB**. Its rebuild took **72.43 s**, reusing the existing ONNX export.

The crop-cache implementation was subsequently changed to transfer uint8 pixels before converting to float32 on the GPU. The current-pipeline results above include this change; the historical table does not isolate its individual speedup.

Half resolution changes model predictions. It is not an equivalent-quality optimization. Observed facial/ear landmark drift needs visual evaluation for the intended use case.

First uncached compilation/warmup took 40.6 s at full resolution and 42.1 s at half resolution. Loading was about 1 s. Startup is excluded from the table. The earlier Mac result, approximately 2700 ms/forward, was user-provided and was not remeasured here.

## Temporal sampling · CPU-only quality tests

Full 1024×768 input, original 720×1280 output, 180 frames at 30 FPS. Replayed the saved every-frame TensorRT predictions on CPU; **no neural-model inference or GPU execution** was used for these quality tests. Both output videos retain all 180 frames and the six-second duration.

| Target inference FPS | Model evaluations needed | Interpolated frames | Fewer evaluations | Median linear deviation | p95 linear deviation | Median hold deviation | p95 hold deviation | Visible-point coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 61 | 119 | 66.1% | 5.61 px | 18.73 px | 11.98 px | 48.50 px | 98.03% |
| 15 | 91 | 89 | 49.4% | 3.48 px | 14.87 px | 8.70 px | 33.35 px | 98.63% |

The final frame is explicitly inferred, explaining 61/91 instead of 60/90. Deviations are in original-image pixels, measured only on interpolated frames with mutually visible/confident reference and interpolated points. Hold and linear use the same comparison points within each test. The 10/15 FPS tests have different interpolated-frame sets. Coverage is the fraction of reference-visible points retained on interpolated frames. Every-frame predictions are a reference, **not ground truth**.

| Landmark group | 10 FPS linear median / p95 | 15 FPS linear median / p95 |
|---|---:|---:|
| Body | 5.70 / 18.52 px | 4.07 / 13.13 px |
| Face | 5.22 / 13.89 px | 3.25 / 9.69 px |
| Hands | 16.77 / 93.99 px | 11.56 / 95.97 px |
| Feet | 7.04 / 27.85 px | 5.88 / 22.37 px |

Recommendation: **15 FPS + linear interpolation** is the better optional general-purpose compromise on this clip. 10 FPS saves more model evaluations; neither mode reliably reconstructs fast hand motion. Full resolution/every-frame inference stays the default. CPU replay, including decoding, quality analysis, drawing, and both output encoders, took 1.25 s at 10 FPS and 1.41 s at 15 FPS; these numbers **exclude neural inference and are not end-to-end GPU throughput**. H.264 browser-preview conversion is excluded too.

An initial 10 FPS GPU test reported 10.977/10.951 s for two 180-frame runs (16.42 output FPS), but a concurrent ComfyUI process occupied approximately 20 GB VRAM and 100% GPU utilization. That result is **contended and excluded from accepted speed comparisons**. GPU testing was paused at the user's request, then resumed with permission once the GPU was idle. See the verified current-pipeline results at the top of this file.

CPU tests cover irregular and 29.97 FPS timestamps, first/final anchors, frame retention, weak/off-frame endpoint suppression, output ordering, and propagation of encoder failures.

Raw CPU replay summaries and videos are saved in `runs/cpu-replay-10/` and `runs/cpu-replay-15/` in the workstation checkout. Sampled `--benchmark` runs measure two complete pipeline runs and skip the separate forward-only microbenchmark.

## Community research

- [Rerun's BF16 static-batch Sapiens2 TensorRT integration](https://github.com/rerun-io/examples-monorepo/pull/38).
- [TensorRT 11 BF16 transposed-convolution fixes and parity checks](https://github.com/rerun-io/examples-monorepo/pull/106).

Their measurements are not substituted for measurements on this workstation. This project uses its own HF-to-ONNX-to-TensorRT code.
