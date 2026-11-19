"""PNG noise (CVPR 2026): real iPhone 7 sRGB sensor noise on images, from the official prompt-driven noise generator"""
import argparse
import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cv_tools import MODEL_DIR, ROOT

PNG_DIR = MODEL_DIR / "png-cvpr2026"
# [Agent] SIDD validation pairs shot on an iPhone 7: the capture's ISO, the SIDD patch, and the blend the lab picked for a
# subtle phone look (D at 28 %). The ISO names the reference capture; nothing here changes exposure.
PROFILES = {"A": (1000, "0024-0002", 0.20), "B": (1250, "0019-0010", 0.30), "C": (1600, "0004-0010", 0.45), "D": (2000, "0014-0011", 0.28)}
TILE, STRIDE, HALO = 256, 192, 32
LOOKS = ("full", "blend", "shadow", "chroma", "night")


def load_model(device):
    """PNG's generator, prompt encoder and decoder from its Lightning checkpoint, without Lightning, plus the sampling constants."""
    sys.path.insert(0, str(ROOT / "third_party" / "png"))
    from png.utils.common import instantiate_from_config

    checkpoint = torch.load(PNG_DIR / "prompt-cm.ckpt", map_location="cpu", weights_only=False)
    config, weights = checkpoint["hyper_parameters"], checkpoint["state_dict"]
    nets = {}
    # [Agent] Sampling runs the EMA copy of the generator, as PNG's own karras_sample does; the training copy is skipped.
    for name, net_config in [("model_ema", config["model_config"]), ("encoder", config["ae_config"]["encoder_config"]),
                             ("decoder", config["ae_config"]["decoder_config"])]:
        net = instantiate_from_config(net_config)
        net.load_state_dict({k.removeprefix(name + "."): v for k, v in weights.items() if k.startswith(name + ".")})
        # [Agent] FP32 on purpose: the three networks are small (0.4 s for a 736x1312 finish on a 3090), and FP32 keeps the
        # noise identical to the finish that was validated by eye.
        nets[name] = net.to(device).eval()
    cm, dit = config["cm_config"], config["model_config"]["params"]
    scale = np.float32(cm["sigma_data"]) / np.float32(cm["ae_raw_std"])
    # [Agent] One consistency step from sigma_max. The schedule's float32 round trip makes sigma a hair off 80, and the
    # boundary scalings below are PNG's, so the noise matches its sampler exactly.
    rho = cm["rho"]
    sigma = (cm["sigma_min"] ** (1 / rho) + torch.ones(1) * (cm["sigma_max"] ** (1 / rho) - cm["sigma_min"] ** (1 / rho))) ** rho
    data, low = cm["sigma_data"], cm["sigma_min"]
    return types.SimpleNamespace(
        generator=nets["model_ema"], encoder=nets["encoder"], decoder=nets["decoder"], device=device,
        latent=(dit["in_channels"], dit["input_size"], dit["input_size"]), sigma_max=cm["sigma_max"], sigma=sigma.to(device),
        c_skip=(data**2 / ((sigma - low) ** 2 + data**2)).to(device), c_out=((sigma - low) * data / (sigma**2 + data**2) ** 0.5).to(device),
        c_in=(1 / (sigma**2 + data**2) ** 0.5).to(device),
        # [Agent] The latent is standardised per channel; decoding undoes it.
        scale=torch.from_numpy(scale).to(device).view(1, -1, 1, 1),
        bias=torch.from_numpy(np.float32(cm["mu_data"]) - np.float32(cm["ae_raw_mean"]) * scale).to(device).view(1, -1, 1, 1))


def get_prompt(model, profile):
    """The noise prompt for one profile: the encoder reads the noise of a real noisy/clean SIDD pair."""
    _, patch, _ = PROFILES[profile]
    noisy, clean = (torch.from_numpy(np.asarray(Image.open(PNG_DIR / "sidd-val" / "validation" / folder / f"{patch}.png").convert("RGB"), np.float32) / 255)
                    for folder in ("input", "groundtruth"))
    out0, out1, out2, out3 = model.encoder((noisy - clean).permute(2, 0, 1)[None].to(model.device) * 2)
    return [F.pixel_unshuffle(out0[1][1], 8), F.pixel_unshuffle(out0[1][0], 8), F.pixel_unshuffle(out1[1][0], 4),
            F.pixel_unshuffle(out2[1][0], 2), out3[1][0]]


def sample_noise(model, image, prompt, seed, batch):
    """The noise residual for a float RGB image [H, W, 3] in 0-1."""
    # [Agent] The model works on 256 px tiles. Tiles overlap by 64 px with a cosine ramp, and the overlap sum is divided
    # by sqrt of the summed squared weights, so seams keep the same noise variance as tile centres. Each tile has its own
    # generator (seed + tile index), so the noise does not depend on --batch.
    h, w = image.shape[:2]
    rows, cols = (h + STRIDE - 1) // STRIDE, (w + STRIDE - 1) // STRIDE
    padded = np.pad(image, ((HALO, rows * STRIDE - h + HALO), (HALO, cols * STRIDE - w + HALO), (0, 0)), mode="reflect")
    ramp = np.sin(np.linspace(0, np.pi / 2, 64, dtype=np.float32)) ** 2
    edge = np.ones(TILE, np.float32)
    edge[:64], edge[-64:] = ramp, ramp[::-1]
    window = (edge[:, None] * edge[None, :])[:, :, None]
    total, weight = np.zeros_like(padded), np.zeros(padded.shape[:2] + (1,), np.float32)
    areas = [np.s_[r * STRIDE:r * STRIDE + TILE, c * STRIDE:c * STRIDE + TILE] for r in range(rows) for c in range(cols)]
    for start in range(0, len(areas), batch):
        chunk = range(start, min(start + batch, len(areas)))
        patches = np.stack([padded[areas[i]] for i in chunk])
        clean = torch.from_numpy(patches).permute(0, 3, 1, 2).to(model.device) * 2 - 1
        condition = F.pixel_unshuffle(F.interpolate(clean, scale_factor=0.5, mode="bicubic", antialias=True), 4)
        x = torch.cat([torch.randn((1, *model.latent), generator=torch.Generator().manual_seed(seed + i)) for i in chunk]).to(model.device)
        x = x * model.sigma_max
        prompts = [p.expand(len(chunk), *p.shape[1:]) for p in prompt] + [condition]
        z = model.c_out * model.generator(model.c_in * x, model.sigma.expand(len(chunk)), prompts) + model.c_skip * x
        noisy = ((model.decoder((z - model.bias) / model.scale, clean) + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        for j, i in enumerate(chunk):
            total[areas[i]] += (noisy[j] - patches[j]) * window
            weight[areas[i]] += window * window
    return (total / np.sqrt(np.maximum(weight, 1e-8)))[HALO:HALO + h, HALO:HALO + w]


def mix_looks(image, noise, blend):
    # [Agent] full = the noise as sampled. The rest are the lab's finishing variants: a subtle blend, more noise in shadows
    # than on lit skin, luma noise with little colour speckle, and both for night shots.
    luma = lambda a: np.einsum("...c,c->...", a, [0.2126, 0.7152, 0.0722])[..., None]  # noqa: E731
    s = 1 - np.clip((luma(image) - 0.03) / 0.42, 0, 1)
    shadow = s * s * (3 - 2 * s)
    y = luma(noise)
    return {"full": image + noise, "blend": image + noise * blend, "shadow": image + noise * (0.18 + 0.52 * shadow),
            "chroma": image + 0.55 * (y + 0.20 * (noise - y)), "night": image + (y + 0.35 * (noise - y)) * (0.15 + 0.38 * shadow)}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, nargs="+", required=True, help="Images")
    p.add_argument("--output", type=Path, help="Fresh run directory; default: runs/png_noise/<timestamp>")
    p.add_argument("--profiles", default="ABCD", help="iPhone 7 references to sample: A=ISO 1000, B=1250, C=1600, D=2000 (default: ABCD)")
    p.add_argument("--looks", nargs="+", choices=LOOKS, default=list(LOOKS), help="Outputs per profile (default: all)")
    p.add_argument("--seed", type=int, default=2601801)
    p.add_argument("--batch", type=int, default=16, help="Tiles per model call (default: 16; lower it if VRAM is short)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if set(args.profiles) - set(PROFILES):
        p.error(f"--profiles takes letters from {''.join(PROFILES)}")
    out = (args.output or ROOT / "runs" / "png_noise" / datetime.now().strftime("%Y%m%d-%H%M%S")).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        p.error(f"Output is not empty: {out}; choose a fresh run directory")
    model = load_model(args.device)
    prompts = {k: get_prompt(model, k) for k in args.profiles}
    for path in args.input:
        image = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255
        # [Agent] Several inputs share a run (the model loads once): each gets <grandparent>__<parent>/, since renders often share file names.
        destination = out / f"{path.parent.parent.name}__{path.parent.name}" if len(args.input) > 1 else out
        destination.mkdir(parents=True, exist_ok=True)
        for k in args.profiles:
            iso, _, blend = PROFILES[k]
            started = time.perf_counter()
            noise = sample_noise(model, image, prompts[k], args.seed, args.batch)
            seconds = time.perf_counter() - started
            for name, result in mix_looks(image, noise, blend).items():
                if name in args.looks:
                    Image.fromarray(np.round(np.clip(result, 0, 1) * 255).astype(np.uint8)).save(destination / f"{path.stem}-{k}{iso}-{name}.png")
            print(f"{path.parent.name}/{path.name} · {k} ISO {iso} · noise std {noise.std() * 255:.1f} DN · {seconds:.2f} s", flush=True)
    (out / "summary.json").write_text(json.dumps({"inputs": [str(x) for x in args.input], "profiles": {k: PROFILES[k] for k in args.profiles},
                                                  "looks": args.looks, "seed": args.seed, "device": args.device, "tile": [TILE, STRIDE, HALO]}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
