# cv-tools

Local image/video inference. Every tool is one Python file you run directly; the model tools share the same flags and output layout.

| Run | What | Output |
|---|---|---|
| `python -m cv_tools.sapiens2.pose` | [Meta Sapiens2 0.8B](https://github.com/facebookresearch/sapiens2) pose | 308 landmarks, skeleton overlay (Sapiens or DWPose look) |
| `python -m cv_tools.sam31` | [Meta SAM 3.1](https://github.com/facebookresearch/sam3) tracker | one object tracked both ways from clicks or a mask; fastest SAM path on the 3090 |
| `python -m cv_tools.motion_guide` | Sapiens2 pose + SAM 3.1 | the H3 `<Video 1>` guide from a clip: the performer blurred inside her mask, her pose drawn on top |
| `python -m cv_tools.shots` | ffmpeg scene score | the hard cuts in a clip: `shots.json` (shot start/end frame and `MM:SS.mmm`) and one frame strip per shot, for writing one-shot or multi-shot prompts |
| `python -m cv_tools.char_sheet` | Sapiens2 pose | a three-column character sheet cut into front body, back body and head, at full resolution |
| `python -m cv_tools.palette` | k-means in Lab | dominant colours, highlight and shadow tint, printed to the terminal |
| `python -m cv_tools.color_match` | quantile curves on background pixels, or a Lab shift inside a mask | an ffmpeg `curves` filter that gives a render its source clip's colour grade, or the render with one region (the face) toned to the source's |

Every file takes `--help`.

## Install

Linux needs an NVIDIA driver for CUDA 13.2; macOS runs on MPS. Python 3.12+.

```bash
git clone https://github.com/FrancescoSaverioZuppichini/cv-tools && cd cv-tools
bash setup.sh                 # one .venv for every tool, plus Meta's SAM 3 code in third_party/
source .venv/bin/activate
```

No `python3 -m venv` on the machine (Ubuntu without `python3-venv`)? Give setup another Python 3.12: `PYTHON=$(uv python find 3.12) bash setup.sh`. `VENV=/path` puts the environment elsewhere.

One environment runs every tool: PyTorch 2.15 (CUDA 13.2 wheels), Transformers 5.23 and, on Linux, TensorRT 11.5. Meta's [SAM 3](https://github.com/facebookresearch/sam3) installs without its pins (it pins `numpy<2`, which would break the other tools).

### Download the weights

Nothing downloads by itself: every tool loads its checkpoint offline from `.models/`. Download what your tools need, from the cv-tools folder with the venv active (it has `hf` and `uvx`). `CV_TOOLS_DATA=/path` moves `.models/` and `engines/` elsewhere (e.g. a data drive); then download into `$CV_TOOLS_DATA/.models/...`.

| Tool | Weights |
|---|---|
| `motion_guide` | Sapiens2 pose 0.8b + SAM 3.1 |
| `char_sheet`, `sapiens2.pose` | Sapiens2 pose 0.8b |
| `sam31` | SAM 3.1 |
| `bars` | Sapiens2 pose 0.8b, only when a shot has bars |
| `shots`, `palette`, `color_match` | nothing |

```bash
# Sapiens2 pose 0.8b (3.4 GB). The repo's sapiens2_0.8b_pose.safetensors is the same weights again: skip it.
hf download facebook/sapiens2-pose-0.8b config.json preprocessor_config.json model.safetensors --local-dir .models/sapiens2-pose-0.8b

# SAM 3.1 (1.7 GB): ComfyUI's sam3.1_multiplex_fp16.safetensors holds the same tensors under Meta's names; re-save it in Meta's layout.
# (Or the gated facebook/sam3.1 download: sam3.1_multiplex.pt into .models/sam3.1/.)
mkdir -p .models/sam3.1
python -c "import torch; from safetensors.torch import load_file; torch.save({'model': load_file('<ComfyUI>/models/checkpoints/sam3.1_multiplex_fp16.safetensors')}, '.models/sam3.1/sam3.1_multiplex.pt')"

```

The first CUDA run of a Sapiens2 size builds its TensorRT engine into `engines/` (a couple of minutes, once per GPU).

## Quick start

```bash
source .venv/bin/activate

python -m cv_tools.sapiens2.pose --input clip.mp4
python -m cv_tools.palette --input photo.jpg
python -m cv_tools.sam31 --input clip.mp4 --frame 141 --points 199,40 203,525
python -m cv_tools.motion_guide --input clip.mp4
python -m cv_tools.char_sheet --input sheet.png
```

## Model tools: shared flags and outputs

`sapiens2.pose`, `sam31` and `motion_guide` take these. The other tools have their own, listed in their sections.

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

GPU inference is half precision (fp16) by default in every model tool. Where fp16 broke an output, the comment at that line says which part stays fp32 and why.

Speeds in this README were measured on the previous PyTorch 2.11 / TensorRT 10 stack. Outputs on the current stack match it: pose within 0.07 px median, SAM 3.1 masks at IoU 1.0 ([`BENCHMARK.md`](cv_tools/sapiens2/BENCHMARK.md#pytorch-215--tensorrt-11-upgrade-check)).

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
| `--model 0.4b\|0.8b\|1b\|5b\|PATH` | Checkpoint size, read from `.models/sapiens2-pose-<size>`, or a local checkpoint; default `0.8b`. A size's TensorRT engine builds on first use (a couple of minutes) |

Outputs `landmarks.jsonl` (original-space xy, heatmap-peak scores, `keypoint_ids`, `null` for invalid points), `overlay.(jpg|mp4)` (pose on the source) and `guide.(png|mp4)` (skeleton on black).

**Backend.** CUDA runs a static, strongly typed FP16 TensorRT engine: FP32 stays exactly where the HF model upcasts (RMSNorm, the attention softmax); MPS runs PyTorch FP16, CPU PyTorch FP32. Preprocessing is HF's crop and normalization as one GPU `grid_sample`, and HF's DARK sub-pixel refinement runs on the GPU too. The first CUDA run per resolution builds `engines/<model>-<HxW>-trt<version>.plan` (minutes, reported as `engine_build_seconds`); later runs load it in about a second. Engines are specific to the GPU, driver and TensorRT version: build on the target machine, and a TensorRT upgrade builds fresh ones.

**Sampling.** `--inference-fps` samples on decoded timestamps (fractional and variable FPS included), always infers the first and final frames, and fills the rest by linear interpolation. Interpolated points take the weaker endpoint score; low-confidence, non-finite or off-image endpoints are dropped rather than bridged. It looks ahead to the next inferred frame, so it is an offline pipeline, not live tracking. JSONL marks each frame `inferred` or `interpolated` with its `anchor_frames`.

**Speed**, idle RTX 3090, 720×1280 clip, full resolution: **17.21 FPS** every frame, **26.64 FPS** at `--inference-fps 15`, **31.75 FPS** at 10. About 1.7 GiB VRAM. 15 FPS stays closer to every-frame predictions (3.48 px median vs 5.61 px at 10); keep every frame for fast hands, cuts or precision. Each refactor was checked against a recorded run: identical landmarks, then 0.07 px median across the TensorRT 11 upgrade. Methodology and history: [`BENCHMARK.md`](cv_tools/sapiens2/BENCHMARK.md).

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

Outputs `masks.npz` (`masks`: bool `[frames, H, W]`, lossless), `masks.jsonl` (per-frame lossless COCO RLE: `from pycocotools import mask; mask.decode(mask.frPyObjects(rle, *rle["size"]))`), `overlay.mp4`, `masks.mp4`.

From Python: `load_tracker()`, `read_video(path, limit, size)` and `track(model, frames, prompt_frame, prompts, stride=...)` return the masks without writing anything, one `[frames, H, W]` per prompt (`{"points": [...]}` or `{"mask": ...}`). Each object gets its own tracker state and the states step through the clip together, so the image encoder runs once per frame for all of them; each mask is bit-identical to tracking it alone. `motion_guide` uses them this way.

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
python -m cv_tools.motion_guide --input selfie.mp4 --mask-prompt "head, hair" --mask-grow 0.02 --include head,face   # face swap: head lines + brow/mouth dots only
python -m cv_tools.motion_guide --input reel.mp4 --mask-prompt person --exclude legs   # waist-up clip: no leg lines
python -m cv_tools.motion_guide --input reel.mp4 --include head,face,hands   # head, face dots and hands, no body skeleton
```

Video and CUDA only; `--benchmark` is refused (`summary.json` already times every stage).

Use it to prepare a source clip for an H3 character swap. The performer is hidden under a heavy blur (so H3 cannot copy her identity or clothes), and her motion is drawn on top: the OpenPose-style body and hand skeleton, plus white dots on the eyebrows and mouth for expression. It is the recipe of the H3 swap that worked (run 28), in one process with no ComfyUI.

1. **Decode once.** Every later step reads the same frames.
2. **Shots.** Hard cuts are found as PySceneDetect's AdaptiveDetector finds them, at its defaults: a frame whose mean HSV change is at least 15 levels and 3x the average of the 2 frames on each side, shots at least 15 frames long. Each shot is tracked on its own, so SAM's memory of one shot never carries her into the next one, or the next one's subject back.
3. **Prompt.** In each shot Sapiens2 pose (TensorRT FP16) checks every 6th frame from its start; the first frame with a person becomes the prompt frame (or `--prompt-frame N`, for its shot), and her joints there become the SAM clicks.
4. **Track.** SAM 3.1 runs from that frame, both directions, stride 2. Frames where it loses her are prompted again the same way, each run of lost frames on its own, until no new prompt frame is left. Frames where pose finds nobody (black, a fade to black, an empty shot) keep an empty mask and go out as in the source; the run prints them and `summary.json` lists them under `empty_frames`. With `--mask-prompt`, SAM 3's text mask on the prompt frame seeds the track instead of the clicks: the instances her joints fall inside, unioned (a part prompt like `head, hair` returns one instance per part). A frame where no instance has one of her joints (a blended frame mid-dissolve, someone else) is passed over for the next one, so the track never starts from the clicks or a stranger's mask instead. The SAM 3 detector comes from the SAM 3.1 checkpoint, so no gated download; it is freed before SAM 3.1 loads (re-prompts after a loss hold both).
5. **Pose, blur and draw on every frame.**
   - Pose runs inside her mask's box, so other people never get her skeleton.
   - The neck is the shoulders' midpoint, as in DWPose.
   - Face dots are thinned to DWPose's 68-point density: 29 dots (5 per brow, the 15-point outer lip ring and the inner lip's 4 corners and centres) against DWPose's 30. Denser clusters get painted into H3's output.
   - The blur is applied inside the mask grown by `--mask-grow` (default 1 % of the long side) with a 2 px feather.
   - The blur, mask growth and feather run on the GPU (torch + torchvision): 58 dB to the OpenCV version, a few ms per frame.
   - Only the drawing's bright pixels are pasted, as in the original recipe, so there is no dark fringe.
   - `--include` / `--exclude` pick the parts drawn: `head` (nose, eyes, ears), `face` (the dots `--face` picks), `torso` (neck, shoulders, hips), `arms`, `hands`, `legs`. A shoulder or hip belongs to two parts and is drawn when either is in; a limb needs both its joints. A face swap draws only `head,face`: the body stays visible, so a skeleton over it adds nothing H3 cannot see and is what it can paint into the render. `keypoints.npz` keeps every point either way.
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
| `--include PARTS` | Draw only these parts, comma-separated: `head`, `face`, `torso`, `arms`, `hands`, `legs` (default: all). Face swap: `head,face` |
| `--exclude PARTS` | Draw every part but these, e.g. `legs`; not with `--include` |
| `--color` | Tint the blur per region: face green, hair red, the rest of her blue; see below |
| `--prompt-every 6` | Frames between pose checks when choosing the SAM prompt frame |
| `--prompt-frame N` | Prompt SAM on frame N (her visible joints there become the clicks) instead of searching its shot |
| `--threshold 0.3` | Keypoint score to trust and draw |

Outputs `guide.mp4` (the `<Video 1>` to feed H3), `masks.npz`, `keypoints.npz` (`points [frames, 308, 2]`, `scores`, `names`) and `summary.json` (prompt frame, clicks, mask prompt, seconds per stage, peak VRAM).

**`--color`** mixes a flat tint into the blur (55 %, before the drawing goes on), so H3 can tell her parts apart under it: face green, hair red, the rest of her blue. A region is tinted only when its part is drawn: `face` → face, `head` → hair, `torso`/`arms`/`hands`/`legs` → body. Face and hair are SAM 3 `face` / `hair` tracks, prompted per shot and re-prompted like the main mask, each in its own tracker state stepping with the main one, so the image encoder still runs once. `masks.npz` keeps the blur mask as `masks` and adds the `face` and `hair` tracks (what `color_match --inside` reads); without the flag the guide is unchanged.

```bash
python -m cv_tools.motion_guide --input clip.mp4 --mask-prompt person --mask-grow 0 --color   # face swap: --mask-prompt "head, hair" --include head,face --color
```

RTX 4090, upright-full (428 frames, 1024×576): 44.6 s → 50.4 s with `--color` (track 13.8 → 19.5 s, peak VRAM 6.1 → 7.9 GB); the old flow (three `motion_guide` runs + a tinting pass) took 145 s.

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

## Colour match · give a render its source's colours

```bash
python -m cv_tools.color_match --source clip.mp4 --render swap.mp4 --masks guide/masks.npz --end 7.3
# stderr: color_match: 29 samples, 783032 px, largest shift r 9, g 14, b 37 levels
# stdout: curves=r='0.0000/0.0000 0.0625/...':g='...':b='...'
ffmpeg -i swap.mp4 -vf "$(python -m cv_tools.color_match ... )" -c:a copy matched.mp4
```

| Flag | Meaning |
|---|---|
| `--source`, `--render` | The clip and the render made from it, on the same clock (the render may have another size or frame rate) |
| `--masks` | motion_guide's `masks.npz` for the source: the person is left out of the fit; without it every pixel counts |
| `--start 0`, `--end` | Seconds sampled; leave out dissolves and fades |
| `--step 0.25` | Seconds between samples |
| `--width 320` | Sample width; height follows the source |
| `--grow 18` | Pixels the person mask grows at sample width, so the new person's hair and arms stay out |
| `--strength 1` | `0.5` moves every curve point halfway back to no change: when the render's background is already close, half keeps the person natural. Render both and pick by eye |

For each of R, G and B, the render's brightness levels at 17 even quantiles are mapped to the source's on the background pixels (person mask grown, letterbox bars and clipped highlights skipped), and read back as a 17-point curve that never goes down. One global curve per channel: a whole-frame tint or grade matches; local light and saturation changes that mix channels do not. **Largest shift** is how far each curve strays from no change, in 0–255 levels: 12–17 on a render whose colour was fine (Coffee time), 36–38 on one that lost the source's green grade (crashing).

**Speed**, 4090 box, 8 s 960×720 clip + 1120×864 render, 29 samples: **0.65 s** including Python start-up (the first version seeked with one ffmpeg call per sample: 6.1 s, same curves within the fit's own noise). CPU only.

**Inside a mask** (a face swap's skin to the source's):

```bash
python -m cv_tools.motion_guide --input clip.mp4 --output guide --color        # masks.npz gets face and hair
python -m cv_tools.color_match --source clip.mp4 --render swap.mp4 --masks guide/masks.npz --inside face --out swap-toned.mp4
# stderr: color_match: face measured on 243/243 frames, shift red +2.9..+4.5, yellow +2.4..+4.0 (Lab a*, b*)
```

| Flag | Meaning |
|---|---|
| `--inside face` | The `masks.npz` key to match inside (`face`, `hair`, `masks`) |
| `--out PATH` | The matched render; its sound is copied |
| `--strength 1` | Scales the shift |

Every render frame is paired with the source frame at the same time. Both are measured inside the source's mask (shrunk 15 px, so hair and background at its edge stay out), on colourful pixels only: skin counts; grey or ash hair, brows, lashes and eye whites do not. The difference of the mean Lab a\* and b\* (red, yellow) is the frame's shift. A moving median over 0.5 s keeps a light change and drops flicker. The shift goes back on the mask, feathered 41 px and weighted the same way, so brightness stays the render's and the pixels outside go to the encoder unchanged (measured drift outside the face: 0.03 Lab, from x264 alone). Decoding and encoding use ffmpeg's exact rounding with BT.709 tags; its default rounding darkened every pixel by about 2 levels on the round trip.

**Speed**, RTX 3090 box, 10 s 736×1312 render: **14 s** including Python start-up. CPU only.

## Bars · take the black bars out, shot by shot

```bash
python -m cv_tools.bars --input clip.mp4 --output out.mp4
# shot 1  00:00.000 - 00:08.367  picture 960x720 at 0,0  no bars
# shot 2  00:08.367 - 00:15.633  picture 416x720 at 272,0  crop 416x312 at 272,112
```

Removes letterbox, pillarbox and a picture inside the picture, and keeps the frame size and the sound (H.264 CRF 12, yuv420p). The clip is cut into shots with `shots.py`'s hard-cut detector, then again wherever the bars change without a cut (a dissolve into a narrow picture is invisible to the detector). Per shot, the picture is the union of every frame's lit box (a pixel above 7 in any channel; rows and columns count when 10 % of them are lit), so a dark scene is never mistaken for bars. A shot with no bars passes through; otherwise the largest box with the frame's aspect inside the picture is cropped, centred on the head (Sapiens2 nose, eyes and ears over 16 frames; the picture's middle when nobody is found) and scaled back with lanczos. Every shot keeps the frame size, so the shots join. GPU for the pose, only when a shot has bars. Prints, per shot, the time range, the picture found and the crop used.

## Layout

```text
cv_tools/
  cli.py              Shared flags, device selection, summary.json
  media.py            Frame reading with timestamps and --size resizing; ordered output worker writing JSONL, images, videos
  tensorrt.py         ONNX export to a static, strongly typed FP16 engine; engine runtime
  sam31.py            SAM 3.1 tracker: load_tracker(), read_video(), track() for other tools; its CLI is run()/main()
  motion_guide.py     Sapiens2 pose + SAM 3.1 + blur + drawing in one pass: the H3 <Video 1> guide
  palette.py          Lab k-means palette, highlight/shadow tint
  color_match.py      A render's colour to its source's: background curves as an ffmpeg filter, or a per-frame shift inside a mask
  bars.py             Black-bar removal per shot: lit-box union, head-centred crop, scale back to the frame size
  sapiens2/
    pose.py           GPU crop, TensorRT/PyTorch forward, DARK refinement, Sapiens and DWPose drawing
    temporal.py       Timestamp sampling, confidence-gated interpolation
    keypoints.json    Meta landmark names, colours, skeleton links
    BENCHMARK.md      RTX 3090 measurements
third_party/sam3/     Meta's SAM 3 / 3.1 code (cloned by setup.sh, installed --no-deps)
```

A new model tool is one file: a `main()` that adds its own flags to an `ArgumentParser`, calls `cli.parse(parser, name)` for the shared ones, and passes the result to its `run(args)`.

```bash
python -m unittest discover -s tests
```

The inference, export and CLI code is our own on top of Transformers' model implementations and the official SAM 3 code; it does not import community inference repositories. Sapiens2 landmark metadata derives from Meta's pose dataset configuration and keeps its license.

