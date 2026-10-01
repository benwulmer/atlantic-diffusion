import os
import re
import glob
import torch
from diffusers import UNet2DModel

import config

ARCH = {
    "out_channels": config.NUM_CHANNELS,
    "layers_per_block": 4,
    "block_out_channels": (256, 256, 320, 320, 384),
    "down_block_types": (
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    "up_block_types": (
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    "attention_head_dim": 64,
    "resnet_time_scale_shift": "scale_shift",
}
IN_CHANNELS = 2 * config.NUM_CHANNELS + 1

CONV_IN_SHAPE = (256, IN_CHANNELS, 3, 3)
N_STATE_DICT_KEYS = 828
PARAM_COUNT_M = 161.4

PRIOR_CONV_IN = (256, 24, 3, 3)
V1_CONV_IN = (64, 24, 3, 3)

# Older checkpoint shapes are kept for mismatch diagnostics.
KNOWN_CONV_IN = {
    (
        256,
        25,
        3,
        3,
    ): "PUBLISHED land-mask + 3-attention UNet, in=2C+1, 828 keys, 161.4M",
    (
        256,
        24,
        3,
        3,
    ): "pre-land-mask 'prior' UNet, in=2C, attention x2, 738 keys, 157.7M",
    (
        128,
        24,
        3,
        3,
    ): "'realwindmodel' UNet, block_out (128,...), in=2C, 742 keys, 160.3M",
    (
        64,
        24,
        3,
        3,
    ): "v1 x0 UNet, block_out (64,128,256,512,512), lpb=2, 392 keys, 99.5M",
}

CKPT_GLOBS = ("allvars*ema*.pt", "allvars*diffusion*.pt", "allvars_v1*.pt")
_EPOCH_RE = re.compile(r"epoch_(\d+)")
_UNNAMED = "the architecture this script builds"


# Build the training U-Net for an (H, W) grid.
def build_unet(H, W, in_channels=None, device=None):
    model = UNet2DModel(
        sample_size=(H, W),
        in_channels=IN_CHANNELS if in_channels is None else int(in_channels),
        **ARCH,
    )
    return model if device is None else model.to(device)


# Summarise the architecture for logs.
def describe():
    return (
        f"UNet2DModel  in={IN_CHANNELS} (2C+1, C={config.NUM_CHANNELS})  "
        f"out={ARCH['out_channels']}\n"
        f"  layers_per_block={ARCH['layers_per_block']}  "
        f"block_out_channels={ARCH['block_out_channels']}\n"
        f"  down={tuple(b.replace('Block2D', '') for b in ARCH['down_block_types'])}\n"
        f"  up  ={tuple(b.replace('Block2D', '') for b in ARCH['up_block_types'])}\n"
        f"  attention_head_dim={ARCH['attention_head_dim']}  "
        f"resnet_time_scale_shift={ARCH['resnet_time_scale_shift']!r}\n"
        f"  expects conv_in.weight {CONV_IN_SHAPE}, {N_STATE_DICT_KEYS} state-dict keys, "
        f"~{PARAM_COUNT_M}M params"
    )


# Read the input-convolution shape and key count, using meta tensors if possible.
def ckpt_fingerprint(path):
    try:
        state = torch.load(path, map_location="meta", weights_only=True, mmap=True)
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=True)
    w = state.get("conv_in.weight")
    shape = tuple(int(s) for s in w.shape) if w is not None else None
    return shape, len(state)


# '<basename>: conv_in (...), N keys  [-> known family]' for error messages.
def describe_ckpt(path):
    shape, n_keys = ckpt_fingerprint(path)
    known = KNOWN_CONV_IN.get(shape)
    tail = f"  -> {known}" if known else "  -> UNRECOGNISED architecture"
    return f"{os.path.basename(path)}: conv_in {shape}, {n_keys} keys{tail}"


# Check the input-convolution shape before loading a checkpoint.
def check_ckpt(ckpt_path, in_channels=None, expect_conv_in=None):
    want = expect_conv_in
    if want is None:
        want = (
            CONV_IN_SHAPE
            if in_channels is None
            else (CONV_IN_SHAPE[0], int(in_channels), 3, 3)
        )
    shape, n_keys = ckpt_fingerprint(ckpt_path)
    if shape != tuple(want):
        raise SystemExit(
            f"\nCheckpoint / architecture mismatch — refusing to load.\n"
            f"  wanted : conv_in.weight {tuple(want)}  "
            f"({KNOWN_CONV_IN.get(tuple(want), _UNNAMED)})\n"
            f"  on disk: {describe_ckpt(ckpt_path)}\n"
            f"  file   : {ckpt_path}\n"
            f"Point CKPT at a checkpoint of the SAME architecture, or run the script that "
            f"builds the architecture above.\nThe published model is:\n{describe()}"
        )
    return shape, n_keys


# Check architecture compatibility and load the full state dict strictly.
def load_checkpoint(model, ckpt_path, device, in_channels=None, expect_conv_in=None):
    check_ckpt(ckpt_path, in_channels=in_channels, expect_conv_in=expect_conv_in)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


# Prefer EMA weights, then final checkpoints, then the latest epoch.
def _rank(path):
    name = os.path.basename(path)
    m = _EPOCH_RE.search(name)
    return ("ema" in name, "final" in name, int(m.group(1)) if m else -1)


# Resolve an explicit checkpoint or search the fallback directories.
def resolve_ckpt(ckpt, fallback_dirs, expect_conv_in=None, verbose=True):
    want = tuple(CONV_IN_SHAPE if expect_conv_in is None else expect_conv_in)
    if ckpt and os.path.exists(ckpt):
        check_ckpt(ckpt, expect_conv_in=want)
        return ckpt

    cands = sorted(
        {
            p
            for d in fallback_dirs
            for g in CKPT_GLOBS
            for p in glob.glob(os.path.join(d, g))
        }
    )
    if not cands:
        raise SystemExit(
            f"No checkpoint at {ckpt!r} and no {' / '.join(CKPT_GLOBS)} in {list(fallback_dirs)}.\n"
            f"scp the EMA checkpoint down from the box first."
        )

    matching, rejected = [], []
    for p in cands:
        try:
            shape, _ = ckpt_fingerprint(p)
        except Exception as e:
            rejected.append(
                f"    {os.path.basename(p)}: unreadable ({type(e).__name__}: {e})"
            )
            continue
        (matching if shape == want else rejected).append(
            p if shape == want else f"    {describe_ckpt(p)}"
        )
    if not matching:
        raise SystemExit(
            f"No checkpoint in {list(fallback_dirs)} has conv_in.weight {want}\n"
            f"({KNOWN_CONV_IN.get(want, _UNNAMED)}).\n"
            f"  Found instead:\n"
            + "\n".join(rejected)
            + f"\nEither point CKPT at the right file or run the script for that architecture."
        )

    best = max(matching, key=_rank)
    if verbose:
        print(
            f"  {ckpt} missing; using the newest ARCHITECTURE-MATCHING checkpoint found: "
            f"{best}\n    {describe_ckpt(best)}"
        )
    return best


if __name__ == "__main__":

    print(describe())
    found = sorted(
        {
            p
            for d in (
                "box_backup",
                "checkpoints/unet_allvars_corrdiff",
                "gpu_allvars_corrdiff",
            )
            for g in CKPT_GLOBS
            for p in glob.glob(os.path.join(d, g))
        }
    )
    if not found:
        print("\nNo checkpoints found to audit.")
    for p in found:
        shape, n_keys = ckpt_fingerprint(p)
        flag = "LOADS " if shape == CONV_IN_SHAPE else "  --  "
        print(f"{flag} {describe_ckpt(p)}")
