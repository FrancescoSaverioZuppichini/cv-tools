# cv-tools

Local image/video inference. Every tool is one Python file you run directly; the model tools share the same flags and output layout.

| Run | What | Output |
|---|---|---|
| `python -m cv_tools.sapiens2.pose` | [Meta Sapiens2 0.4B](https://github.com/facebookresearch/sapiens2) pose | 308 landmarks, skeleton overlay (Sapiens or DWPose look) |
| `python -m cv_tools.sapiens2.contour` | [Meta Sapiens2 0.4B](https://huggingface.co/facebook/sapiens2-seg-0.4b) body-part segmentation | per-part outlines on black, label maps |
| `python -m cv_tools.depth_anything` | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) small/base/large | relative depth map |
| `python -m cv_tools.sam3` | [Meta SAM 3](https://huggingface.co/facebook/sam3) | text-prompted instance masks, tracked across video frames |
| `python -m cv_tools.sam31` | [Meta SAM 3.1](https://github.com/facebookresearch/sam3) tracker | one object tracked both ways from clicks or a mask; fastest SAM path on the 3090 |
| `python -m cv_tools.motion_guide` | Sapiens2 pose + SAM 3.1 | the H3 `<Video 1>` guide from a clip: the performer blurred inside her mask, her pose drawn on top |
| `python -m cv_tools.char_sheet` | Sapiens2 pose | a three-column character sheet cut into front body, back body and head, at full resolution |
| `python -m cv_tools.palette` | k-means in Lab | dominant colours, highlight and shadow tint, printed to the terminal |
| `python -m cv_tools.canny` | OpenCV Canny, auto thresholds | edge map (white on black), image or video |
| `python -m cv_tools.png_noise` | [PNG](https://github.com/JK-the-Ko/PNG) (CVPR 2026) noise generator | real iPhone 7 sensor noise on images, five finishing looks |
| `python -m cv_tools.phone_finish` | sharpen + PNG grain + JPEG | renders that read like phone photos |
| `python -m cv_tools.social` | phone ISP, Instagram, WhatsApp compression | what a photo looks like after it has been posted and re-sent |

Every file takes `--help`.

## Quick start

```bash
source .venv/bin/activate

python -m cv_tools.sapiens2.pose --input clip.mp4
python -m cv_tools.sapiens2.contour --input photo.jpg --output runs/contour/photo
python -m cv_tools.depth_anything --input photo.jpg
python -m cv_tools.sam3 --input clip.mp4 --prompt person "shopping bag"
python -m cv_tools.palette --input photo.jpg
python -m cv_tools.canny --input photo.jpg
python -m cv_tools.sam31 --input clip.mp4 --frame 141 --points 199,40 203,525
python -m cv_tools.motion_guide --input clip.mp4
python -m cv_tools.char_sheet --input sheet.png
python -m cv_tools.phone_finish --input renders/
```

## Model tools: shared flags and outputs

`sapiens2.pose`, `depth_anything`, `sam3`, `sam31` and `motion_guide` take these. `contour`, `palette`, `canny`, `png_noise`, `phone_finish` and `social` have their own, listed in their sections.

| Flag | Meaning |
|---|---|
| `--input PATH` | Image (`jpg png webp bmp tif`) or video; required |
| `--output DIR` | Fresh run directory; default `runs/<tool>/<timestamp>/`; a non-empty directory is refused |
| `--frames N` | Only the first N frames; default all, or 30 with `--benchmark` |
| `--benchmark` | Two complete measured runs instead of one |
| `--warmup N` | Warmup passes on the first frame before timing; default 5 |
| `--size 0.5\|1mp\|720x1280` | Resize input frames before the model: a scale, megapixels (aspect kept) or exact width×height. Outputs come out at that size |
| `--device auto\|cuda\|mps\|cpu` | `auto` is CUDA on Linux, MPS on macOS. A missing GPU fails instead of silently using CPU |

Every run writes:

```text
summary.json          Environment, settings, load/warmup time, per-run timing, FPS, GPU memory
run-1/
  <records>.jsonl     One line per frame: index, source and output timestamps, the model's result
  <images or videos>  Per tool, below
run-2/                --benchmark only
```

GPU inference is half precision (fp16) by default in every model tool. Where fp16 broke an output, the comment at that line says which part stays fp32 and why. PNG noise runs in fp32: its networks are small (0.4 s for a 736×1312 phone finish) and fp32 keeps the noise it was validated with.

Speeds in this README were measured on the previous PyTorch 2.11 / TensorRT 10 stack. Outputs on the current stack match it: pose within 0.07 px median, SAM 3.1 masks at IoU 1.0, depth within 0.2 % ([`BENCHMARK.md`](cv_tools/sapiens2/BENCHMARK.md#pytorch-215--tensorrt-11-upgrade-check)).

Images give still files; videos give silent `mp4v` MP4s at the source FPS with every frame kept. Drawing and encoding run on one background thread while the GPU predicts the next frame. Timing covers decode, inference, drawing, JSONL and encoding; loading, engine builds and warmup are excluded.

## Sapiens2 · human pose

```bash
python -m cv_tools.sapiens2.pose --input clip.mp4                     # full 1024×768, every frame
python -m cv_tools.sapiens2.pose --input clip.mp4 --inference-fps 10  # infer at 10 FPS, interpolate the rest
python -m cv_tools.sapiens2.pose --input photo.jpg --box 100 80 500 900 --landmarks body hands
python -m cv_tools.sapiens2.pose --input clip.mp4 --style dwpose      # OpenPose-style guide for pose ControlNets
```

| Flag | Meaning |
|---|---|
| `--resolution full\|half\|quarter\|HxW` | Model input, not output size: 1024×768 (default), 512×384, 256×192, or sides divisible by 32 |
| `--box X Y W H` | Fixed person box in source pixels; default the whole frame. No detector or tracker, single person |
| `--landmarks all\|body\|face\|hands\|feet` | Groups to draw/export, combinable; the model still predicts all 308 |
| `--style sapiens\|dwpose` | Drawing. `sapiens` (default): the selected landmarks in Sapiens colours. `dwpose`: the OpenPose-18 body with its fixed per-limb colours, 21-point rainbow hands and white face dots (brows, eyes, nose, lips; Sapiens2 has no jaw line), the picture pose ControlNets are trained on |
| `--threshold 0.3` | Score to draw a point and to trust it as an interpolation endpoint |
| `--inference-fps N` | Infer at N FPS, linearly interpolate the frames between |
| `--model 0.4b\|0.8b\|1b\|5b\|PATH` | Checkpoint size, read from `.models/sapiens2-pose-<size>`, or a local checkpoint; default `0.4b`. A size's TensorRT engine builds on first use (a couple of minutes) |

Outputs `landmarks.jsonl` (original-space xy, heatmap-peak scores, `keypoint_ids`, `null` for invalid points), `overlay.(jpg|mp4)` (pose on the source) and `guide.(png|mp4)` (skeleton on black).

**Backend.** CUDA runs a static, strongly typed FP16 TensorRT engine: FP32 stays exactly where the HF model upcasts (RMSNorm, the attention softmax); MPS runs PyTorch FP16, CPU PyTorch FP32. Preprocessing is HF's crop and normalization as one GPU `grid_sample`, and HF's DARK sub-pixel refinement runs on the GPU too. The first CUDA run per resolution builds `engines/<model>-<HxW>-trt<version>.plan` (minutes, reported as `engine_build_seconds`); later runs load it in about a second. Engines are specific to the GPU, driver and TensorRT version: build on the target machine, and a TensorRT upgrade builds fresh ones.

**Sampling.** `--inference-fps` samples on decoded timestamps (fractional and variable FPS included), always infers the first and final frames, and fills the rest by linear interpolation. Interpolated points take the weaker endpoint score; low-confidence, non-finite or off-image endpoints are dropped rather than bridged. It looks ahead to the next inferred frame, so it is an offline pipeline, not live tracking. JSONL marks each frame `inferred` or `interpolated` with its `anchor_frames`.

**Speed**, idle RTX 3090, 720×1280 clip, full resolution: **17.21 FPS** every frame, **26.64 FPS** at `--inference-fps 15`, **31.75 FPS** at 10. About 1.7 GiB VRAM. 15 FPS stays closer to every-frame predictions (3.48 px median vs 5.61 px at 10); keep every frame for fast hands, cuts or precision. Each refactor was checked against a recorded run: identical landmarks, then 0.07 px median across the TensorRT 11 upgrade. Methodology and history: [`BENCHMARK.md`](cv_tools/sapiens2/BENCHMARK.md).

## Sapiens2 · body-part contours

```bash
python -m cv_tools.sapiens2.contour --input photo.jpg --output runs/contour/photo
python -m cv_tools.sapiens2.contour --input photos/ --output runs/contour/batch --min-area 0.002
```

| Flag | Meaning |
|---|---|
| `--input PATH` | An image, or a folder: every `jpg jpeg png webp` directly inside; required |
| `--output DIR` | Created if missing; required |
| `--model PATH` | Default `.models/sapiens2-seg-0.4b` |
| `--min-area 0.0005` | Drop outline fragments smaller than this share of the image |

Per image: `<name>.png` (each part's outline, 2 px, in its own colour on black) and `<name>-labels.png` (8-bit label map, 0 = background, 1–28 the parts). One `parts.json` for the run: per image and label, `pixels` and `centroid_xy` (0–1). Parts stay unnamed (`label_<n>`): the checkpoint ships only `LABEL_n`, and the Sapiens v1 (Goliath) names do not match its order. Images only, CUDA FP16, none of the shared flags.

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

Images and videos run through one `Sam3VideoModel` streaming session: frames go in one at a time, object IDs stay stable across a video, and memory does not grow with clip length. Streaming skips SAM 3's hotstart filter, which needs future frames, so an object seen for only a few frames is kept rather than retroactively removed. FP16 on GPU, FP32 on CPU. **Not yet run end-to-end: the gated checkpoint is not downloaded.**

## SAM 3.1 · track one object through a video

```bash
python -m cv_tools.sam31 --input clip.mp4 --frame 141 --points 199,40 203,525 257,1050
python -m cv_tools.sam31 --input clip.mp4 --frame 0 --mask first-frame-mask.png --stride 1
```

Use it to get a person's (or any object's) mask on every frame of a clip, from one prompt on one frame. Tracking runs forward and backward from that frame, so the prompt frame can be anywhere in the clip, e.g. the frame where the person is most visible.

| Flag | Meaning |
|---|---|
| `--points X,Y ...` / `--negative X,Y ...` | Positive and negative clicks, in pixels, on `--frame` |
| `--mask PNG` | Instead of clicks: the object's mask on `--frame` (white = object) |
| `--frame N` | Prompt frame (default 0) |
| `--direction both\|forward\|backward` | Default `both` |
| `--stride N` | Default 2: track every 2nd frame; a skipped frame gets the union of its two tracked neighbours (it over-covers a fast move, never cuts a limb). `--stride 1` tracks every frame |
| `--batch 8` | Frames per image-encoder pass, prefetched in the direction of travel |
| `--compile` | `torch.compile` the image encoder: ~2 min once per process, ~8 % faster |
| `--checkpoint PATH` | Default `.models/sam3.1/sam3.1_multiplex.pt` |
| `--alpha 0.5` | Mask opacity in the overlay |

Outputs `masks.npz` (`masks`: bool `[frames, H, W]`, lossless), `masks.jsonl` (per-frame RLE, as in `sam3`), `overlay.mp4`, `masks.mp4`.

From Python: `load_tracker()`, `read_video(path, limit, size)` and `track(model, frames, prompt_frame, points=..., stride=...)` return the masks without writing anything. `motion_guide` uses them this way.

**Backend.**
- Meta's `facebookresearch/sam3` in tracker-only mode. Meta's default video predictor runs the text detector on every frame: 2× slower, and it loses click-only objects.
- fp16 matrix weights; norms and RoPE tables stay fp32. A blanket half-precision cast moved the RoPE positions and dropped mask IoU from 0.99 to 0.95.
- The clip is decoded once into pinned CPU memory, so VRAM does not grow with clip length.
- The encoder runs on batches of 8 frames, computing only the outputs tracking reads.

Masks match ComfyUI's SAM 3.1 node (IoU 0.99 at stride 1, 0.985 at stride 2) at ~2–3× its speed.

**Tried and dropped (RTX 3090):**
- fp16 GEMM accumulation: slower, IoU 0.95.
- Batches beyond 8: +3 %.
- Meta's compile of the tracker heads: CUDA graphs crash with the tracker's memory.
- TensorRT for the encoder:
  - with every fp32-typed layer pinned (Sapiens' policy): correct, but 370 ms/frame;
  - with free FP16/BF16 precision: 66 ms/frame, but broken masks (IoU 0.0007).
  
  Even fixed, it would gain ~12 % on the encoder, because PyTorch already uses fused cutlass GEMMs and flash attention here.

## Motion guide · the H3 `<Video 1>` guide from a clip

```bash
python -m cv_tools.motion_guide --input reel.mp4 --output runs/motion_guide/reel
python -m cv_tools.motion_guide --input reel.mp4 --size 0.5 --face none
python -m cv_tools.motion_guide --input reel.mp4 --pose-model 0.8b   # bigger Sapiens2 pose; its engine builds on first use
python -m cv_tools.motion_guide --input reel.mp4 --mask-prompt person --mask-grow 0.05   # SAM 3 text mask as the seed, wider blur
```

Video and CUDA only; `--benchmark` is refused (`summary.json` already times every stage).

Use it to prepare a source clip for an H3 character swap. The performer is hidden under a heavy blur (so H3 cannot copy her identity or clothes), and her motion is drawn on top: the OpenPose-style body and hand skeleton, plus white dots on the eyebrows and mouth for expression. It is the recipe of the H3 swap that worked (run 28), in one process with no ComfyUI.

1. **Decode once.** Every later step reads the same frames.
2. **Prompt.** Sapiens2 pose (TensorRT FP16) checks every 6th frame from the start; the first frame with a person becomes the prompt frame (or `--prompt-frame N`), and her joints there become the SAM clicks. If SAM loses her on any frame, the run stops and names the frames, so no frame goes out unblurred.
3. **Track.** SAM 3.1 runs from that frame, both directions, stride 2. With `--mask-prompt`, SAM 3's text mask on the prompt frame seeds the track instead of the clicks: the instances her joints fall inside, unioned (a part prompt like `head, hair` returns one instance per part). The SAM 3 detector comes from the SAM 3.1 checkpoint, so no gated download; it is freed before SAM 3.1 loads.
4. **Pose, blur and draw on every frame.**
   - Pose runs inside her mask's box, so other people never get her skeleton.
   - The neck is the shoulders' midpoint, as in DWPose.
   - Face dots are thinned to DWPose's 68-point density: 29 dots (5 per brow, the 15-point outer lip ring and the inner lip's 4 corners and centres) against DWPose's 30. Denser clusters get painted into H3's output.
   - The blur is applied inside the mask grown by `--mask-grow` (default 1 % of the long side) with a 2 px feather.
   - The blur, mask growth and feather run on the GPU (torch + torchvision): 58 dB to the OpenCV version, a few ms per frame.
   - Only the drawing's bright pixels are pasted, as in the original recipe, so there is no dark fringe.
   - Encoding is libx264 CRF 14, on a worker thread with one frame in flight while the GPU runs the next frame's pose.

| Flag | Meaning |
|---|---|
| `--stride N` | SAM stride (default 2) |
| `--pose-model 0.4b\|0.8b\|1b\|5b\|PATH` | Sapiens2 pose checkpoint, as in `sapiens2.pose --model`; default `0.8b` (keeps arms on low selfie angles where 0.4b drops them) |
| `--pose-res full\|half` | Pose model input. `full` (1024×768) is the default. `half` (512×384) makes pose 3.5× faster, but joints land a median 4 px and up to ~11 px from full, mostly fast hands |
| `--mask-prompt TEXT` | Seed the track from SAM 3's mask for this text instead of the joint clicks; comma-separated phrases are pooled. `person` keeps her clothes in; `head, hair` for a face-only swap (her body and clothes stay visible) |
| `--mask-grow 0.01` | Blur area grown beyond the mask, as a fraction of the long side; wider hides what sticks out of the mask (flyaway hair, straps) |
| `--blur-div 15` | Blur sigma = longest side / this; 15 is run 28's heavy blur |
| `--face brows_mouth\|all\|none` | Face dots (default eyebrows + mouth) |
| `--prompt-every 6` | Frames between pose checks when choosing the SAM prompt frame |
| `--prompt-frame N` | Prompt SAM on frame N (her visible joints there become the clicks) instead of searching |
| `--threshold 0.3` | Keypoint score to trust and draw |

Outputs `guide.mp4` (the `<Video 1>` to feed H3), `masks.npz`, `keypoints.npz` (`points [frames, 308, 2]`, `scores`, `names`) and `summary.json` (prompt frame, clicks, mask prompt, seconds per stage, peak VRAM).

**Benchmark vs the ComfyUI pipeline it replaces**: `persontrack.py` (DWPose + MediaPipe + SAM 3.1 node, run twice) then `vidprep2.py` (DWPose again, CPU blur, drawing). RTX 3090, same clips, back to back, 13 Nov 2026.

| Clip | ComfyUI | motion_guide | Speed-up | Peak VRAM ComfyUI / motion_guide | Masks vs ComfyUI's |
|---|---|---|---|---|---|
| reel01, 201 frames 736×1312 | 517.6 s (114.6 + 403.0) | **41.3 s** | **12.5×** | 3.0 / 7.5 GB | IoU 0.988, her pixels covered 99.7 % (1 frame < 98 %) |
| bridge, 250 frames 736×1312 | 652.0 s (147.6 + 504.4) | **47.2 s** | **13.8×** | 3.1 / 7.5 GB | IoU 0.955, covered 99.1 %; misses frame 1 (motion-blurred head in the whip pan) |

`motion_guide` times include Python start and model loading. Its stages on reel01, after the clean-up and speed pass: decode 0.4 s, load pose 0.6 s, find prompt 2.0 s, load SAM 4.1 s (random init skipped), track 12.7 s, pose + draw + encode 12.0 s. Total **31.8 s**, or ~23 s with `--pose-res half`.

Where the old pipeline lost its time:
- **The blur:** a sigma-87 Gaussian on the CPU, 910 ms per frame. Here it runs on the frame shrunk 8× and scaled back: 3.5 ms, PSNR 48 dB to the exact blur.
- **DWPose run twice:** the pose ran once in the tracker and again for the drawing.
- **MediaPipe:** run only for comparison.
- **ComfyUI round trips:** per-frame PNGs.

**Known gaps:**
- No re-prompting yet. The old tracker re-prompted failing stretches; this costs the bridge's first entrance frame.
- Peak VRAM is higher: SAM's 8-frame encoder batch plus PyTorch's cache, with the pose engine loaded.

## Character sheet · cut a three-column sheet into its parts

```bash
python -m cv_tools.char_sheet --input sheet.png
python -m cv_tools.char_sheet --input sheet.png --output runs/char_sheet/s1004 --grow 1.2 --floor 0.05
```

Image and CUDA only. The input is a character reference sheet with three columns split by thin white gaps: the full body from the front, the full body from the back, and a column whose top half is a face close-up (the bottom half is ignored). It only crops: nothing is redrawn and every crop keeps the input's full resolution, so upscale the sheet first if you want bigger crops.

1. **Columns.** The gaps are the columns that stay whiter than the studio background from top to bottom. The face panel's divider is often too faint to find; the third column is then split in half.
2. **Bodies.** Sapiens2 pose on each body column. Both bodies get the same two cuts, so front and back keep the same scale: just under the chin (`--below-chin` of the chin-to-shoulders distance: the head goes, the neck and shoulders stay) and just under the lowest foot point (`--floor` of the body height). The run stops if the top cut would clip the back view's shoulders.
3. **Head.** Sapiens2's face landmarks frame the box: from `--hair` of the brows-to-chin span above the brows to `--under-chin` below the chin, `--aspect` wide, grown by `--grow` around its centre. It is centred between the eyes and kept symmetric inside the panel, so the face is always in the middle.

| Flag | Meaning |
|---|---|
| `--pose-model 0.8b` | Sapiens2 pose size (`0.4b`, `0.8b`, `1b`, `5b`) or a checkpoint path |
| `--score 0.3` | Keypoint confidence to trust |
| `--below-chin 0.1` | Body top cut under the chin, as a share of chin-to-shoulders |
| `--floor 0.03` | Floor kept under the lowest foot point, as a share of the body height |
| `--hair 0.75` | Head top above the brows, as a share of brows-to-chin |
| `--under-chin 0.12` | Head bottom under the chin, as a share of brows-to-chin |
| `--aspect 0.8` | Head width over height |
| `--grow 1.12` | Head box scale around its centre (1 = the tight box) |

Of the shared flags, `--input`, `--output` and `--device` apply.

Outputs `front.png`, `back.png` and `head.png` at full resolution, `proof.jpg` (the sheet with every pose point and cut drawn) and `summary.json` (the columns, chin, shoulders, lowest foot point, the cut rows, the head box, how far the face sits from its centre, and the settings).

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

## Canny · edge map

```bash
python -m cv_tools.canny --input photo.jpg                    # auto thresholds from the median brightness
python -m cv_tools.canny --input clip.mp4 --low 50 --high 150 # fixed thresholds
```

| Flag | Meaning |
|---|---|
| `--input PATH` / `--output DIR` | As in the shared flags; default output `runs/canny/<timestamp>/` |
| `--low N` / `--high N` | Hysteresis thresholds; default auto: (1 ± sigma) × median brightness |
| `--sigma 0.33` | Spread of the auto thresholds around the median |
| `--blur 5` | Odd Gaussian kernel before edge detection, so skin and fabric texture do not become speckle; `0` disables |
| `--frames N` | Frame limit for videos; default all |
| `--size 0.5\|1mp\|720x1280` | Resize input frames first, as in the shared flags |

Outputs `canny.png` (image) or `canny.mp4` (video, source FPS) and `edges.jsonl` (per frame: the thresholds used and the edge pixel count). CPU only, no model download.

## PNG noise · real phone sensor noise

```bash
python -m cv_tools.png_noise --input render.png                                  # 4 iPhone 7 profiles × 5 looks
python -m cv_tools.png_noise --input a.png b.png --profiles D --looks blend shadow
```

| Flag | Meaning |
|---|---|
| `--input PATH...` | Images; required |
| `--output DIR` | Fresh run directory; default `runs/png_noise/<timestamp>/` |
| `--profiles ABCD` | Noise references to sample, real SIDD iPhone 7 captures: A = ISO 1000, B = 1250, C = 1600, D = 2000 |
| `--looks ...` | Outputs per profile, default all: `full` (the noise as sampled), `blend` (at the profile's subtle strength), `shadow` (more noise in shadows than on lit skin), `chroma` (luma noise, little colour speckle), `night` (both) |
| `--seed 2601801` | Same seed, same noise |
| `--batch 16` | 256 px tiles per model call; lower it if VRAM is short |
| `--device cuda\|cpu` | Default CUDA when available |

Outputs `<name>-<profile><ISO>-<look>.png` and `summary.json`; several inputs get one `<grandparent>__<parent>/` folder each, since renders share file names.

[PNG](https://github.com/JK-the-Ko/PNG) learned real sRGB camera noise from noisy/clean pairs: its encoder turns the noise of a real pair into a prompt, and a one-step consistency model samples new noise in that style for any clean picture. Sampling runs on 256 px tiles that overlap by 64 px with a cosine ramp; the overlap is divided by the root of the summed squared weights, so seams keep the variance of tile centres. Each tile has its own seed (`seed + tile index`), so `--batch` never changes the noise. The networks are built from PNG's Lightning checkpoint without Lightning and sampled with the same maths as its `karras_sample`, in the same environment as every other tool.

## Phone finish · sharpen, grain, JPEG

```bash
python -m cv_tools.phone_finish --input renders/                   # every image inside -> renders-finished/
python -m cv_tools.phone_finish --input a.png b.png --output out/ --profile C --blend 0.4
```

Makes a render read like a phone photo: an unsharp mask, then real iPhone 7 ISO 2000 grain at 28 %, saved as JPEG 95 with full-resolution colour at the same size. The grain is sampled for the clean picture and added on top of the sharpened one, so the grain itself is not sharpened. Use it on final images, not on training data: a LoRA trained on finished pictures learns the grain as part of the subject.

| Flag | Meaning |
|---|---|
| `--input PATH...` | Images and/or folders (every image directly inside); required |
| `--output DIR` | Default `<input folder>-finished/` next to it |
| `--sharpen 1.0` | Unsharp amount at radius 1.2 px; `0` is off |
| `--profile A\|B\|C\|D` | Noise reference, as in `png_noise`; default `D` (ISO 2000) |
| `--blend` | Noise strength 0–1; default the profile's subtle blend (D = 0.28) |
| `--quality 95` | JPEG quality |
| `--max-side N` | Shrink (Lanczos) so the long side is at most N first |
| `--look-scale` | Finish as if the picture were this many times smaller: grain size and sharpen radius grow with it. Default long side / 1536, at least 1, so a 2× upscale gets the same visible finish as the render it came from |
| `--seed`, `--batch`, `--device` | As in `png_noise` |

## Social · the trip a phone photo takes

```bash
python -m cv_tools.social --input render.png
python -m cv_tools.social --input finished.jpg --no-phone      # already carries real phone noise
```

Three JPEGs per image in `<output>/<grandparent>/<parent>/` (default output `runs/social/<timestamp>/`): `-phone.jpg` the phone's own processing (chroma denoise, strong ISP sharpening, JPEG 90), `-ig.jpg` Instagram (1080 px wide, box downscale, JPEG 78) and `-wa.jpg` a WhatsApp re-send of that (JPEG 70), all 4:2:0 like the real thing. `--no-phone` skips the phone stage for images that already carry real phone noise (`png_noise`, `phone_finish`): denoising and sharpening them again would erase it. CPU only.

## Installation

Linux: an NVIDIA driver supporting CUDA 13.2. macOS: MPS-capable hardware. Python 3.12+.

```bash
bash setup.sh             # PYTHON=/path/to/python3.12 picks the interpreter, VENV=/path puts the environment elsewhere
source .venv/bin/activate
```

One environment runs every tool: PyTorch 2.15 (CUDA 13.2 wheels), Transformers 5.23 and, on Linux, TensorRT 11.5. `setup.sh` also clones two research repos into `third_party/`: Meta's [SAM 3](https://github.com/facebookresearch/sam3), installed without its pins (it pins `numpy<2`, which would break the other tools), and [PNG](https://github.com/JK-the-Ko/PNG), imported from its folder.

Checkpoints live in `.models/` and load offline; nothing downloads at run time. `CV_TOOLS_DATA=/path` moves `.models/` and `engines/` to that folder (e.g. a data drive); the download commands below then take `$CV_TOOLS_DATA/.models/...`.

```bash
hf download facebook/sapiens2-pose-0.4b --local-dir .models/sapiens2-pose-0.4b
hf download facebook/sapiens2-pose-0.8b --local-dir .models/sapiens2-pose-0.8b   # optional: --model / --pose-model 0.8b
hf download facebook/sapiens2-seg-0.4b --local-dir .models/sapiens2-seg-0.4b     # sapiens2.contour
hf download depth-anything/Depth-Anything-V2-Small-hf --local-dir .models/depth-anything-v2-small
hf download depth-anything/Depth-Anything-V2-Base-hf --local-dir .models/depth-anything-v2-base
hf download depth-anything/Depth-Anything-V2-Large-hf --local-dir .models/depth-anything-v2-large
# SAM 3 is gated: accept the license at huggingface.co/facebook/sam3, run `hf auth login`, then
hf download facebook/sam3 --local-dir .models/sam3 --exclude sam3.pt
```

`sam3.pt` is the 3.4 GB original-format duplicate of `model.safetensors`; the tool does not need it.

SAM 3.1 (`sam31`, `motion_guide`) and PNG (`png_noise`, `phone_finish`) weights:

```bash
# SAM 3.1: ComfyUI's sam3.1_multiplex_fp16.safetensors holds the same tensors under Meta's names; re-save it in Meta's layout.
# (Or the gated facebook/sam3.1 download, sam3.1_multiplex.pt.)
.venv/bin/python -c "import torch; from safetensors.torch import load_file; torch.save({'model': load_file('<ComfyUI>/models/checkpoints/sam3.1_multiplex_fp16.safetensors')}, '.models/sam3.1/sam3.1_multiplex.pt')"
# PNG: the generator checkpoint, and the SIDD validation pairs its noise references come from
.venv/bin/uvx gdown 1Ry46H1BjgKGPLKCNMj89VtmS9ouRHWUS -O .models/png-cvpr2026/prompt-cm.ckpt
.venv/bin/uvx gdown 18qzVfQlEM9fP8wOb9GH28ZEgC8dBT7-N -O sidd-val.tar
mkdir -p .models/png-cvpr2026/sidd-val && tar -xf sidd-val.tar -C .models/png-cvpr2026/sidd-val && rm sidd-val.tar
```

## Layout

```text
cv_tools/
  cli.py              Shared flags, device selection, summary.json
  media.py            Frame reading with timestamps and --size resizing; ordered output worker writing JSONL, images, videos
  tensorrt.py         ONNX export to a static, strongly typed FP16 engine; engine runtime
  depth_anything.py   GPU preprocessing, depth, colormaps
  sam3.py             Streaming segmentation/tracking, overlay, RLE masks
  sam31.py            SAM 3.1 tracker: load_tracker(), read_video(), track() for other tools; its CLI is run()/main()
  motion_guide.py     Sapiens2 pose + SAM 3.1 + blur + drawing in one pass: the H3 <Video 1> guide
  palette.py          Lab k-means palette, highlight/shadow tint
  canny.py            Auto-threshold Canny edges
  png_noise.py        PNG's networks without Lightning: tiled one-step noise sampling, finishing looks
  phone_finish.py     Sharpen + PNG grain + JPEG
  social.py           Phone, Instagram and WhatsApp compression
  sapiens2/
    pose.py           GPU crop, TensorRT/PyTorch forward, DARK refinement, Sapiens and DWPose drawing
    contour.py        Body-part segmentation, per-part outlines
    temporal.py       Timestamp sampling, confidence-gated interpolation
    keypoints.json    Meta landmark names, colours, skeleton links
    BENCHMARK.md      RTX 3090 measurements
third_party/sam3/     Meta's SAM 3 / 3.1 code (cloned by setup.sh, installed --no-deps)
third_party/png/      PNG's networks (cloned by setup.sh, imported from the folder)
```

A new model tool is one file: a `main()` that adds its own flags to an `ArgumentParser`, calls `cli.parse(parser, name)` for the shared ones, and passes the result to its `run(args)`.

```bash
python -m unittest discover -s tests
```

The inference, export and CLI code is our own on top of Transformers' model implementations and the official SAM 3 and PNG code; it does not import community inference repositories. Sapiens2 landmark metadata derives from Meta's pose dataset configuration and keeps its license.

