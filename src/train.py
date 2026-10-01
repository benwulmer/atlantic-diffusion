import glob
import json
import time
import math
import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.training_utils import compute_snr

import config
import baseline_nodes
import noise as noise_mod

USE_CUDA = torch.cuda.is_available()

if not USE_CUDA and os.environ.get("ALLOW_CPU") != "1":
    raise SystemExit(
        "CUDA is not available — torch.cuda.is_available() is False.\n"
        "  PyTorch sees no GPU; check the NVIDIA driver and that this is a CUDA\n"
        '  build of torch (python -c "import torch; print(torch.version.cuda)").\n'
        "  Set ALLOW_CPU=1 to override for a CPU-only smoke test."
    )
# torchrun supplies rank and device placement.
DISTRIBUTED = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
if DISTRIBUTED:
    dist.init_process_group(backend="nccl" if USE_CUDA else "gloo")
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    if USE_CUDA:
        torch.cuda.set_device(LOCAL_RANK)
    DEVICE = f"cuda:{LOCAL_RANK}" if USE_CUDA else "cpu"
else:
    RANK, WORLD_SIZE, LOCAL_RANK = 0, 1, 0
    DEVICE = "cuda" if USE_CUDA else "cpu"
IS_MAIN = RANK == 0


# Print only from the main process.
def log(*a, **k):
    if IS_MAIN:
        print(*a, **k, flush=True)


log("Starting")
log(f"Distributed: {DISTRIBUTED} | world size: {WORLD_SIZE} | device: {DEVICE}")
if USE_CUDA:
    log(
        f"GPU: {torch.cuda.get_device_name(LOCAL_RANK)}  "
        f"VRAM: {torch.cuda.get_device_properties(LOCAL_RANK).total_memory/1e9:.1f}GB"
    )
    torch.backends.cudnn.benchmark = True


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


EPOCHS = _env_int("EPOCHS", 60)
# Batch size and worker count are per GPU.
BATCH_SIZE = _env_int("BATCH_SIZE", 4)

NUM_WORKERS = _env_int("NUM_WORKERS", 4)

SMOKE_ARCH = os.environ.get("SMOKE_ARCH") == "1"
GRADIENT_CHECKPOINTING = True
TIMESTEPS = 1000
LR = 2e-4
LR_MIN = 1e-6
WARMUP_STEPS = 500
SNR_GAMMA = 5.0
OUT_FOLDER = os.environ.get("OUT_FOLDER", "gpu_allvars_corrdiff")
SEED = _env_int("SEED", 0)

# Historical variable name: this weights the L1 term, not an MSE loss.
MSE_WEIGHT = 0.9
DIST_WEIGHT = 0.1
DIST_BINS = 96
DIST_RANGE = (-10.0, 10.0)
DIST_ABAR_MIN = 0.01

# Zero selects uniform timestep sampling, as used for the final model.
HIGH_T_FRACTION = float(os.environ.get("HIGH_T_FRACTION", 0.0))

PREDICTION_TYPE = os.environ.get("PREDICTION_TYPE", "v_prediction")
HIGH_T_START = 950

EMA_DECAY = 0.9999
SAVE_EVERY = 10
RESUME = os.environ.get("RESUME") == "1"

C = config.NUM_CHANNELS

torch.manual_seed(SEED + 1000 * RANK)
np.random.seed(SEED + 1000 * RANK)


# Pair normalised truth with a randomly selected SEAS5 member baseline.
class TruthResidualDataset(Dataset):

    def __init__(
        self,
        shard_path,
        nodes_path,
        node_idx,
        node_w,
        valid_sec,
        stats,
        n_members,
        smear_sigma,
        smear_max,
        seed,
    ):
        self.truth = np.load(shard_path, mmap_mode="r")
        self.nodes = np.load(nodes_path, mmap_mode="r")
        self.node_idx = node_idx
        self.node_w = node_w
        self.valid_sec = np.asarray(valid_sec, dtype=np.float64)
        self.n_members = int(n_members)
        self.per_member = self.nodes.ndim == 5
        self.smear_sigma = float(smear_sigma)
        self.smear_max = int(smear_max)
        self.seed = int(seed)
        self._rng = None

        assert len(self.truth) == len(self.node_idx), (
            f"truth shard has {len(self.truth)} rows but stats has "
            f"{len(self.node_idx)} node weights - the shard and stats_allvars.npy "
            f"come from DIFFERENT prepare_data runs; re-run prepare_data.py"
        )
        if self.per_member:
            assert self.nodes.shape[1] == self.n_members, (
                f"nodes carry {self.nodes.shape[1]} members, stats says "
                f"{self.n_members}"
            )

        ts = np.asarray(stats["truth_std"], dtype=np.float32)[:, None, None]
        tm = np.asarray(stats["truth_mean"], dtype=np.float32)[:, None, None]
        bs = np.asarray(stats["base_std"], dtype=np.float32)[:, None, None]
        bm = np.asarray(stats["base_mean"], dtype=np.float32)[:, None, None]
        rs = np.asarray(stats["res_std"], dtype=np.float32)[:, None, None]
        rm = np.asarray(stats["res_mean"], dtype=np.float32)[:, None, None]
        self.a = ts / rs
        self.b = bs / rs
        self.c = (tm - bm - rm) / rs

        self.step_sec = 21600.0

    def __len__(self):
        return self.truth.shape[0]

    # Initialise a separate NumPy generator for each data-loader worker.
    def _rand(self):
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng(self.seed + 7919 * (wid + 1))
        return self._rng

    # Draw a nearby truth row without crossing a time gap or split boundary.
    def _smear_index(self, i, rng):
        if self.smear_sigma <= 0:
            return i
        d = int(round(rng.normal(0.0, self.smear_sigma)))
        d = max(-self.smear_max, min(self.smear_max, d))
        if d == 0:
            return i
        j = i + d
        if j < 0 or j >= len(self.valid_sec):
            return i
        if abs(self.valid_sec[j] - self.valid_sec[i] - d * self.step_sec) > 1.0:
            return i
        return j

    def __getitem__(self, i):
        rng = self._rand()
        m = int(rng.integers(self.n_members)) if self.per_member else 0
        j = self._smear_index(i, rng)

        truth = self.truth[j].astype(np.float32)
        if self.per_member:
            base = baseline_nodes.eval_nodes_member(
                self.nodes, m, self.node_idx[i], self.node_w[i]
            )
        else:
            base = baseline_nodes.eval_nodes(
                self.nodes, self.node_idx[i], self.node_w[i]
            )
        res = self.a * truth - self.b * base + self.c
        return torch.from_numpy(res), torch.from_numpy(base)


# Normalised histogram with Gaussian soft bin assignments.
def soft_histogram(x, bins, x_min, x_max):
    bin_width = (x_max - x_min) / bins
    bin_centers = torch.linspace(
        x_min + bin_width / 2, x_max - bin_width / 2, bins, device=x.device
    )
    x_flat = x.reshape(-1, 1)
    weights = torch.exp(-0.5 * ((x_flat - bin_centers.unsqueeze(0)) / bin_width) ** 2)
    hist = weights.sum(dim=0)
    return hist / (hist.sum() + 1e-8)


# Mean per-channel KL(target || prediction), pooling batch and spatial axes.
def distribution_kld_loss(pred, target, bins=DIST_BINS, val_range=DIST_RANGE):
    x_min, x_max = val_range
    total = torch.zeros((), device=pred.device)
    for c in range(pred.shape[1]):
        pred_hist = soft_histogram(pred[:, c], bins, x_min, x_max)
        tgt_hist = soft_histogram(target[:, c], bins, x_min, x_max)
        total = total + F.kl_div(torch.log(pred_hist + 1e-8), tgt_hist, reduction="sum")
    return total / pred.shape[1]


# Recover the clean residual from the noisy field and network output.
def predict_x0(noisy, model_out, t, alphas_cumprod):
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    if PREDICTION_TYPE == "v_prediction":
        return a_bar.sqrt() * noisy - (1.0 - a_bar).sqrt() * model_out
    return (noisy - (1.0 - a_bar).sqrt() * model_out) / a_bar.sqrt()


# Draw uniform timesteps, or split draws between low and high noise.
def sample_timesteps(batch, device, generator=None):
    if HIGH_T_FRACTION <= 0.0:
        return torch.randint(0, TIMESTEPS, (batch,), device=device)
    hi = torch.rand(batch, device=device) < HIGH_T_FRACTION
    t_lo = torch.randint(0, HIGH_T_START, (batch,), device=device)
    t_hi = torch.randint(HIGH_T_START, TIMESTEPS, (batch,), device=device)
    return torch.where(hi, t_hi, t_lo)


# Exponential moving average of model weights, held on rank zero.
class EMA:

    def __init__(self, model, decay):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd, num_updates=0):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)
        self.num_updates = int(num_updates)


log("Loading stats...")
stats = np.load(config.STATS_PATH, allow_pickle=True).item()
schema = int(stats.get("schema", 1))
if schema < 2:
    raise SystemExit(
        f"stats_allvars.npy is schema {schema}; this trainer needs schema 2 "
        f"(normalised TRUTH shards + truth_mean/truth_std). Re-run "
        f"prepare_data.py, or check out the schema-1 trainer."
    )
H_pad, W_pad = int(stats["H_pad"]), int(stats["W_pad"])
N_MEMBERS = int(stats.get("n_members", 1))
log(f"  Padded spatial: ({H_pad}, {W_pad})  channels: {C}  SEAS5 members: {N_MEMBERS}")

use_member = config.USE_MEMBER_BASELINE and N_MEMBERS > 1
nodes_path = config.NODES_NORM_MEM_PATH if use_member else config.NODES_NORM_PATH
if not os.path.exists(nodes_path):
    raise SystemExit(
        f"{nodes_path} not found - run prepare_data.py "
        f"(USE_MEMBER_BASELINE={config.USE_MEMBER_BASELINE})"
    )
log(
    f"  Baseline nodes: {nodes_path} "
    f"({'per-member' if use_member else 'ensemble mean'})"
)
log(f"  Forward noise: {noise_mod.describe(config)}")
log(
    f"  Temporal smear: sigma={config.TEMPORAL_SMEAR_SIGMA} steps "
    f"(max |offset| {config.TEMPORAL_SMEAR_MAX}) = "
    f"{config.TEMPORAL_SMEAR_SIGMA*6:.1f} h"
)

log("Loading sea mask...")
_sea_np = (
    np.load(config.SEA_MASK_PATH, allow_pickle=True).item()["mask"].astype(np.float32)
)

_dh, _dw = int(stats["H"]), int(stats["W"])
assert _sea_np.shape == (_dh, _dw), (
    f"sea_mask {_sea_np.shape} != data grid ({_dh},{_dw}). sea_mask.npy and "
    f"stats_allvars.npy must come from the SAME prepare_data run (stale file?)."
)
assert (_dh, _dw) == (H_pad, W_pad), (
    f"data grid ({_dh},{_dw}) != model size ({H_pad},{W_pad}); this script assumes "
    f"no padding (grid is a multiple of {config.PAD_MULTIPLE})."
)
land_channel = torch.from_numpy(_sea_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
log(f"  Land channel: (1,1,{_dh},{_dw}), ocean fraction: {_sea_np.mean():.3f}")

dataset = TruthResidualDataset(
    config.split_data_path("train"),
    nodes_path,
    stats["node_idx_train"],
    stats["node_w_train"],
    stats["valid_sec_train"],
    stats,
    N_MEMBERS if use_member else 1,
    config.TEMPORAL_SMEAR_SIGMA,
    config.TEMPORAL_SMEAR_MAX,
    SEED,
)
log(f"  Train samples: {len(dataset)}")

if DISTRIBUTED:
    sampler = DistributedSampler(
        dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, seed=SEED
    )
else:
    sampler = None
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    shuffle=(sampler is None),
    num_workers=NUM_WORKERS,
    pin_memory=(USE_CUDA and NUM_WORKERS > 0),
    persistent_workers=(NUM_WORKERS > 0),
    drop_last=DISTRIBUTED,
)
steps_per_epoch = len(loader)
total_steps = steps_per_epoch * EPOCHS

log("Building model...")
if SMOKE_ARCH:
    log("*** SMOKE_ARCH=1: tiny UNet, NOT checkpoint-compatible with the samplers ***")
    model = UNet2DModel(
        sample_size=(H_pad, W_pad),
        in_channels=2 * C + 1,
        out_channels=C,
        layers_per_block=1,
        block_out_channels=(32, 64),
        norm_num_groups=8,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
        resnet_time_scale_shift="scale_shift",
    ).to(DEVICE)
else:
    model = UNet2DModel(
        sample_size=(H_pad, W_pad),
        in_channels=2 * C + 1,
        out_channels=C,
        layers_per_block=4,
        block_out_channels=(256, 256, 320, 320, 384),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        attention_head_dim=64,
        resnet_time_scale_shift="scale_shift",
    ).to(DEVICE)
log(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
if GRADIENT_CHECKPOINTING:
    model.enable_gradient_checkpointing()
    log("Gradient checkpointing: ON")
if DISTRIBUTED:
    model = DDP(
        model,
        device_ids=[LOCAL_RANK] if USE_CUDA else None,
        output_device=LOCAL_RANK if USE_CUDA else None,
    )
param_dev = next(model.parameters()).device
log(f"Model on device: {param_dev}")
if USE_CUDA:
    assert param_dev.type == "cuda", f"model is on {param_dev}, expected CUDA"

net = model.module if DISTRIBUTED else model
ema = EMA(net, EMA_DECAY) if IS_MAIN else None

scheduler = DDPMScheduler(
    num_train_timesteps=TIMESTEPS,
    beta_schedule="squaredcos_cap_v2",
    clip_sample=False,
    prediction_type=PREDICTION_TYPE,
)
alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)


def _lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return LR_MIN / LR + (1.0 - LR_MIN / LR) * 0.5 * (1 + math.cos(math.pi * progress))


lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
log(
    f"Training for {EPOCHS} epochs ({total_steps} steps/process, "
    f"effective batch {BATCH_SIZE * WORLD_SIZE})"
)
if IS_MAIN:
    os.makedirs(OUT_FOLDER, exist_ok=True)
    os.makedirs(f"checkpoints/{OUT_FOLDER}", exist_ok=True)

HPARAMS = dict(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    world_size=WORLD_SIZE,
    effective_batch=BATCH_SIZE * WORLD_SIZE,
    num_workers=NUM_WORKERS,
    timesteps=TIMESTEPS,
    lr=LR,
    lr_min=LR_MIN,
    warmup_steps=WARMUP_STEPS,
    snr_gamma=SNR_GAMMA,
    mse_weight=MSE_WEIGHT,
    dist_weight=DIST_WEIGHT,
    dist_bins=DIST_BINS,
    dist_range=list(DIST_RANGE),
    dist_abar_min=DIST_ABAR_MIN,
    high_t_fraction=HIGH_T_FRACTION,
    high_t_start=HIGH_T_START,
    ema_decay=EMA_DECAY,
    seed=SEED,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    beta_schedule="squaredcos_cap_v2",
    prediction_type=PREDICTION_TYPE,
    loss=(
        "min-SNR L1 (v form min(SNR,g)/(SNR+1))"
        if PREDICTION_TYPE == "v_prediction"
        else "min-SNR L1 (published form min(SNR,g)/SNR)"
    ),
    schema=schema,
    seas5_members=N_MEMBERS if use_member else 1,
    per_member_baseline=bool(use_member),
    temporal_smear_sigma=config.TEMPORAL_SMEAR_SIGMA,
    temporal_smear_max=config.TEMPORAL_SMEAR_MAX,
    **noise_mod.describe(config),
)


# Write configuration and provenance beside the training checkpoints.
def write_metadata(tag):
    if not IS_MAIN:
        return
    path = os.path.join(OUT_FOLDER, f"run_metadata_{tag}.json")
    try:
        import run_metadata

        run_metadata.write(path, HPARAMS, config)
        return
    except Exception:
        pass
    import platform
    import hashlib

    meta = {
        "tag": tag,
        "hparams": HPARAMS,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "host": platform.node(),
        "cuda": torch.version.cuda,
        "gpus": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if USE_CUDA
            else []
        ),
        "config": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in vars(config).items()
            if k.isupper() and not k.startswith("_")
        },
        "code_sha256": {
            os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()
            for f in sorted(glob.glob("*.py"))
        },
    }
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    log(f"Wrote {path}")


# Save the model, optimizer, scheduler, scaler and EMA state for resuming.
def save_resume(epoch, step):
    if not IS_MAIN:
        return
    path = os.path.join(OUT_FOLDER, "resume_latest.pt")
    tmp = path + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "global_step": step,
            "model": net.state_dict(),
            "ema": ema.state_dict(),
            "ema_updates": ema.num_updates,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "lr_sched": lr_sched.state_dict(),
            "hparams": HPARAMS,
        },
        tmp,
    )
    os.replace(tmp, path)


start_epoch, global_step = 0, 0
if RESUME:
    rpath = os.path.join(OUT_FOLDER, "resume_latest.pt")
    if not os.path.exists(rpath):
        raise SystemExit(f"RESUME=1 but {rpath} does not exist")
    ck = torch.load(rpath, map_location=DEVICE, weights_only=False)
    prev = ck.get("hparams", {})

    for k in (
        "loss",
        "noise_kind",
        "per_member_baseline",
        "schema",
        "effective_batch",
        "epochs",
    ):
        if k in prev and prev[k] != HPARAMS.get(k):
            raise SystemExit(
                f"resume mismatch on {k!r}: checkpoint has {prev[k]!r}, this run "
                f"wants {HPARAMS.get(k)!r}. Start a fresh run instead."
            )
    net.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"])
    lr_sched.load_state_dict(ck["lr_sched"])
    if IS_MAIN:
        ema.load_state_dict(ck["ema"], ck.get("ema_updates", 0))
    start_epoch = int(ck["epoch"]) + 1
    global_step = int(ck["global_step"])
    log(f"Resumed from {rpath}: epoch {start_epoch}, step {global_step}")
elif IS_MAIN and glob.glob(os.path.join(OUT_FOLDER, "*.pt")):
    log(
        f"WARNING: {OUT_FOLDER}/ already holds checkpoints and RESUME is not set. "
        f"They will be OVERWRITTEN at the next save. Move them aside or set RESUME=1."
    )

write_metadata("start")
t_start = time.time()
step_times = []
skipped_steps = 0

log("Starting training")
for epoch in range(start_epoch, EPOCHS):
    if DISTRIBUTED:
        sampler.set_epoch(epoch)
    epoch_loss = 0.0
    for residual, baseline in loader:
        step_start = time.time()
        residual = residual.to(DEVICE, non_blocking=True)
        baseline = baseline.to(DEVICE, non_blocking=True)
        if global_step == 0 and USE_CUDA:
            assert residual.is_cuda and baseline.is_cuda, "batch is not on CUDA"
            log(f"Confirmed: training batches on {residual.device}")

        # Use the same offset-noise law during training and sampling.
        noise = noise_mod.noise_like(residual.shape, DEVICE, cfg=config)
        t = sample_timesteps(residual.shape[0], DEVICE)
        noisy_residual = scheduler.add_noise(residual, noise, t)
        land_b = land_channel.expand(residual.shape[0], -1, -1, -1)
        model_input = torch.cat([noisy_residual, baseline, land_b], dim=1)

        # bfloat16 avoids the activation overflow seen with float16.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_CUDA):
            pred_noise = model(model_input, t).sample
            snr = compute_snr(scheduler, t)

            # The min-SNR denominator depends on the prediction target.
            min_snr = torch.stack([snr, SNR_GAMMA * torch.ones_like(t)], dim=1).min(
                dim=1
            )[0]
            err_w = (
                min_snr / (snr + 1.0)
                if PREDICTION_TYPE == "v_prediction"
                else min_snr / snr
            )

            target = (
                scheduler.get_velocity(residual, noise, t)
                if PREDICTION_TYPE == "v_prediction"
                else noise
            )
            err = F.l1_loss(pred_noise.float(), target.float(), reduction="none")
            err = (err.mean(dim=list(range(1, err.ndim))) * err_w).mean()

            # Apply the histogram term only at well-conditioned timesteps.
            keep = alphas_cumprod[t] > DIST_ABAR_MIN
            if keep.any():
                x0_hat = predict_x0(
                    noisy_residual[keep].float(),
                    pred_noise[keep].float(),
                    t[keep],
                    alphas_cumprod,
                )
                dist_loss = distribution_kld_loss(x0_hat, residual[keep].float())
            else:
                dist_loss = torch.zeros((), device=DEVICE)
            loss = MSE_WEIGHT * err + DIST_WEIGHT * dist_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        prev_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        # Advance the schedule and EMA only if the optimizer step ran.
        stepped = scaler.get_scale() >= prev_scale
        if stepped:
            lr_sched.step()
            if IS_MAIN:
                ema.update(net)
        else:
            skipped_steps += 1

        if scaler.get_scale() < 1.0:
            raise SystemExit(
                f"GradScaler scale collapsed to {scaler.get_scale()} at step "
                f"{global_step} -- losses/grads are persistently non-finite. "
                f"Aborting; a run in this state cannot learn."
            )

        step_times.append(time.time() - step_start)
        epoch_loss += loss.item()
        global_step += 1

        if global_step % 3000 == 0:
            avg_step = np.mean(step_times[-5:])
            remaining = (total_steps - global_step) * avg_step
            hrs, rem = divmod(remaining, 3600)
            mins, secs = divmod(rem, 60)
            mem = (
                f" | VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB peak"
                if USE_CUDA
                else ""
            )

            log(
                f"Step {global_step:6d}/{total_steps} | Epoch {epoch+1:3d}/{EPOCHS} | "
                f"Loss: {loss.item():.5f} | L1: {err.item():.5f} | "
                f"Dist: {dist_loss.item():.5f} | LR: {lr_sched.get_last_lr()[0]:.2e} | "
                f"Step: {avg_step:.2f}s{mem} | skipped: {skipped_steps} | "
                f"ETA: {int(hrs)}h {int(mins)}m {int(secs)}s"
            )

    elapsed = time.time() - t_start
    e_h, e_r = divmod(elapsed, 3600)
    e_m, e_s = divmod(e_r, 60)
    log(
        f"Finished epoch {epoch+1:3d}/{EPOCHS} | Avg loss: {epoch_loss/steps_per_epoch:.5f} | "
        f"Elapsed: {int(e_h)}h {int(e_m)}m {int(e_s)}s\n"
    )
    save_resume(epoch, global_step)
    if (epoch + 1) % SAVE_EVERY == 0 and IS_MAIN:
        torch.save(
            net.state_dict(), f"{OUT_FOLDER}/allvars_diffusion_epoch_{epoch+1}.pt"
        )
        torch.save(ema.state_dict(), f"{OUT_FOLDER}/allvars_ema_epoch_{epoch+1}.pt")
        write_metadata(f"epoch_{epoch+1}")

log(
    f"Training complete in {(time.time()-t_start)/60:.1f} minutes "
    f"({skipped_steps} steps skipped by GradScaler)"
)
if IS_MAIN:
    torch.save(net.state_dict(), f"checkpoints/{OUT_FOLDER}/allvars_diffusion_final.pt")
    torch.save(ema.state_dict(), f"checkpoints/{OUT_FOLDER}/allvars_ema_final.pt")
    write_metadata("final")
    log(
        f"Saved raw + EMA checkpoints to checkpoints/{OUT_FOLDER}/ "
        "(sample from the EMA one: allvars_ema_final.pt)"
    )
if DISTRIBUTED:
    dist.destroy_process_group()
