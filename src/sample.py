import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import glob
import time
import shutil
import collections
import numpy as np
import torch
import torch.multiprocessing as mp
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler

import config
import baseline_nodes
import model_arch
import noise as noise_mod
import run_metadata

CKPT = os.environ.get("CKPT", "checkpoints/unet_allvars_corrdiff/allvars_ema_final.pt")
SPLIT = os.environ.get("SPLIT", "val")
N_TIMES = int(os.environ.get("N_TIMES", 1000))
N_MEMBERS = int(os.environ.get("N_MEMBERS", config.SEAS5_N_MEMBERS))
MEMBER_BATCH = int(os.environ.get("MEMBER_BATCH", 5))
INFERENCE_STEPS = int(os.environ.get("INFERENCE_STEPS", 500))
SOLVER = os.environ.get("SOLVER", "ddim")
ETA = float(os.environ.get("ETA", 0.5))

TIMESTEP_SPACING = "leading"
# This must match the checkpoint training target.
PREDICTION_TYPE = os.environ.get("PREDICTION_TYPE", "v_prediction")

SEED = int(os.environ.get("SEED", 0))
PICK_FILE = os.environ.get("PICK_FILE")

OUT_DIR = os.environ.get("OUT_DIR", f"samples_final_{SPLIT}_s{INFERENCE_STEPS}")
CKPT_DIRS = ("checkpoints/unet_allvars_corrdiff", "gpu_allvars_corrdiff", "box_backup")

if INFERENCE_STEPS > 0 and 1000 % INFERENCE_STEPS != 0:
    raise SystemExit(
        f"INFERENCE_STEPS={INFERENCE_STEPS} does not divide 1000. With any timestep "
        f"spacing that leaves the chain starting away from t=999 and silently "
        f"skipping the highest-noise steps. Use 100/125/200/250/500/1000."
    )


# Select the same sorted date indices in every worker.
def pick_dates(n_split):
    if PICK_FILE:
        pick = np.unique(np.asarray(np.load(PICK_FILE), dtype=int))
        if pick.size == 0:
            raise SystemExit(f"PICK_FILE {PICK_FILE} is empty")
        if pick.min() < 0 or pick.max() >= n_split:
            raise SystemExit(
                f"PICK_FILE indices span [{pick.min()},{pick.max()}], outside the "
                f"{SPLIT} split of size {n_split}"
            )
        return pick
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(n_split, size=min(N_TIMES, n_split), replace=False))


# Map generated members to SEAS5 members, cycling if necessary.
def member_assignment(n_ens, n_seas5):
    if n_seas5 <= 1:
        return np.zeros(n_ens, dtype=np.int16)
    if n_ens > n_seas5:
        print(
            f"  WARNING: {n_ens} ensemble members but only {n_seas5} SEAS5 members; "
            f"{n_ens - n_seas5} baselines are duplicates and the forecast-spread "
            f"contribution is understated. Set N_MEMBERS <= {n_seas5}."
        )
    return (np.arange(n_ens) % n_seas5).astype(np.int16)


# DDIM (or DPM-Solver++) matched to the training schedule.
def build_scheduler():
    kw = dict(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type=PREDICTION_TYPE,
        timestep_spacing=TIMESTEP_SPACING,
    )
    if SOLVER == "dpm":

        return DPMSolverMultistepScheduler(
            solver_order=2, algorithm_type="dpmsolver++", **kw
        )

    # Clip predicted x0 in normalised residual units.
    return DDIMScheduler(clip_sample=True, clip_sample_range=10.0, **kw)


# Zero retains the full schedule for v-prediction.
ABAR_MIN_X0 = float(os.environ.get("ABAR_MIN_X0", 0.0))


# Keep schedule steps whose cumulative alpha meets ABAR_MIN_X0.
def usable_timesteps(sched):
    keep = sched.alphas_cumprod[sched.timesteps] >= ABAR_MIN_X0
    return sched.timesteps[keep]


# Interpolate all members at one time, returning (M, C, H, W) float32.
def eval_nodes_all_members(nodes, idx, w):
    out = nodes[int(idx[0])].astype(np.float32) * w[0]
    for j in range(1, 4):
        out += nodes[int(idx[j])].astype(np.float32) * w[j]
    return out


# Generate the dates assigned to one device.
def run_worker(rank, world_size, ckpt):
    if torch.cuda.is_available():
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
        amp_dt = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        amp_dt = "mps"
    else:
        device = "cpu"
        amp_dt = "cpu"
    use_amp = device != "cpu"

    # Match training precision on CUDA; MPS uses float16.
    amp_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float16

    stats = np.load(config.STATS_PATH, allow_pickle=True).item()
    schema = int(stats.get("schema", 1))
    if schema < 2:
        raise SystemExit("stats_allvars.npy is schema 1; re-run prepare_data.py")
    C = config.NUM_CHANNELS
    H, W = int(stats["H"]), int(stats["W"])
    n_seas5 = int(stats.get("n_members", 1))
    use_member = config.USE_MEMBER_BASELINE and n_seas5 > 1
    nodes_path = config.NODES_NORM_MEM_PATH if use_member else config.NODES_NORM_PATH
    nodes = np.load(nodes_path, mmap_mode="r")
    node_idx = stats[f"node_idx_{SPLIT}"]
    node_w = stats[f"node_w_{SPLIT}"]
    valid_sec = stats[f"valid_sec_{SPLIT}"]

    def col(k):
        return np.asarray(stats[k], dtype=np.float32)[:, None, None]

    res_std, res_mean = col("res_std"), col("res_mean")
    base_std, base_mean = col("base_std"), col("base_mean")
    truth_std, truth_mean = col("truth_std"), col("truth_mean")

    shard_path = config.split_data_path(SPLIT)
    truth_shard = (
        np.load(shard_path, mmap_mode="r") if os.path.exists(shard_path) else None
    )
    if truth_shard is not None:
        assert truth_shard.shape[-2:] == (H, W), (
            f"truth shard {shard_path} is {tuple(truth_shard.shape[-2:])}, expected "
            f"({H},{W}) — stale/wrong-grid shard"
        )
        assert len(truth_shard) == len(node_idx), (
            f"truth shard has {len(truth_shard)} rows but stats has {len(node_idx)} "
            f"for split '{SPLIT}' — shard and stats are from different runs"
        )

    sea = (
        np.load(config.SEA_MASK_PATH, allow_pickle=True)
        .item()["mask"]
        .astype(np.float32)
    )
    assert sea.shape == (
        H,
        W,
    ), f"sea_mask {sea.shape} != data grid ({H},{W}) — stale file?"
    land_channel = torch.from_numpy(sea).unsqueeze(0).unsqueeze(0).to(device)

    model = model_arch.build_unet(H, W, device=device)
    model_arch.load_checkpoint(model, ckpt, device)
    model.eval()

    sched = build_scheduler()
    sched.set_timesteps(INFERENCE_STEPS)
    timesteps = usable_timesteps(sched)
    members = member_assignment(N_MEMBERS, n_seas5 if use_member else 1)

    my_pick = pick_dates(len(node_idx))[rank::world_size]
    tag = f"w{rank}"
    print(
        f"[{tag}] {device}: {len(my_pick)} dates  ({model.config.in_channels}ch in)  "
        f"{len(timesteps)} steps, first t={int(timesteps[0])} "
        f"({len(sched.timesteps) - len(timesteps)} near-singular steps dropped)",
        flush=True,
    )

    t0 = time.time()
    done = 0
    for k, idx in enumerate(my_pick, 1):
        idx = int(idx)
        out_path = os.path.join(OUT_DIR, f"sample_{idx:06d}.npz")
        if os.path.exists(out_path):
            continue
        when = str(
            np.datetime64("1970-01-01") + np.timedelta64(int(valid_sec[idx]), "s")
        )[:16].replace("T", " ")
        eta_s = (
            f"  ETA {(time.time() - t0) / done * (len(my_pick) - k + 1) / 3600:.1f}h"
            if done
            else ""
        )
        print(f"[{tag}] {k}/{len(my_pick)}  {when}  (idx {idx}){eta_s}", flush=True)

        if use_member:
            base_all = eval_nodes_all_members(nodes, node_idx[idx], node_w[idx])
            base_mem = base_all[members]
            base_ref = base_all.mean(axis=0)
        else:
            base_ref = baseline_nodes.eval_nodes(nodes, node_idx[idx], node_w[idx])
            base_mem = np.repeat(base_ref[None], N_MEMBERS, axis=0)

        # Per-date seeds keep results independent of GPU assignment.
        gnd = torch.Generator(device=device).manual_seed(SEED * 1_000_003 + idx)
        base_t = torch.from_numpy(base_mem).to(device)

        chunks = []
        for start in range(0, N_MEMBERS, MEMBER_BATCH):
            mb = min(MEMBER_BATCH, N_MEMBERS - start)
            base_b = base_t[start : start + mb]
            land_b = land_channel.expand(mb, -1, -1, -1)

            x = noise_mod.noise_like((mb, C, H, W), device, generator=gnd, cfg=config)
            for t in timesteps:
                with torch.no_grad():
                    with torch.autocast(
                        device_type=amp_dt, enabled=use_amp, dtype=amp_dtype
                    ):
                        pred = model(torch.cat([x, base_b, land_b], dim=1), t).sample
                    if SOLVER == "dpm":
                        x = sched.step(pred.float(), t, x).prev_sample
                    elif ETA > 0:
                        vn = noise_mod.noise_like(
                            x.shape, device, generator=gnd, cfg=config
                        )
                        x = sched.step(
                            pred.float(), t, x, eta=ETA, variance_noise=vn
                        ).prev_sample
                    else:
                        x = sched.step(pred.float(), t, x, eta=0.0).prev_sample
            chunks.append(x.float().cpu().numpy())
        res_norm = np.concatenate(chunks, axis=0)

        # Reconstruct physical fields, then store in truth normalisation.
        field = (res_norm * res_std + res_mean) + (base_mem * base_std + base_mean)
        norm = (field - truth_mean) / truth_std

        if not np.isfinite(norm).all():
            bad = [
                config.CHANNELS[c]
                for c in range(C)
                if not np.isfinite(norm[:, c]).all()
            ]
            raise SystemExit(
                f"idx {idx}: non-finite generated field in {bad} — the "
                f"checkpoint or the noise settings are wrong, not a "
                f"storage problem."
            )

        peak = float(np.abs(norm).max())
        if peak > 100.0:
            c = int(np.unravel_index(np.abs(norm).argmax(), norm.shape)[1])
            raise SystemExit(
                f"idx {idx}: |field|/truth_std peaks at {peak:.3g} in channel "
                f"{config.CHANNELS[c]}; expected O(1)-O(20). The model is diverging "
                f"or stats_allvars.npy does not match the checkpoint. Refusing to "
                f"write a bank that every downstream metric would silently average."
            )
        samples = norm.astype(np.float16)

        rec = {
            "samples": samples,
            "baseline": base_ref.astype(np.float16),
            "member_ids": members,
            "index": idx,
            "valid_time": np.int64(valid_sec[idx]),
        }
        if truth_shard is not None:
            rec["truth"] = np.asarray(truth_shard[idx], dtype=np.float16)[:, :H, :W]
        tmp = out_path + ".tmp.npz"
        np.savez(tmp, **rec)
        os.replace(tmp, out_path)

        done += 1
    print(f"[{tag}] done: {len(my_pick)} dates", flush=True)


# Check that every selected date has an output file; report missing years.
def check_complete(pick, valid_sec):
    have = {
        int(os.path.basename(f)[7:13])
        for f in glob.glob(os.path.join(OUT_DIR, "sample_*.npz"))
    }
    missing = [int(i) for i in pick if int(i) not in have]
    if not missing:
        print(f"Bank COMPLETE: {len(pick)}/{len(pick)} dates present.")
        return True
    pos = {int(i): j for j, i in enumerate(pick)}
    yrs = collections.Counter(
        int(
            str(
                np.datetime64("1970-01-01")
                + np.timedelta64(int(valid_sec[pos[i]]), "s")
            )[:4]
        )
        for i in missing
    )
    print(f"\n*** BANK INCOMPLETE: {len(missing)} of {len(pick)} dates are MISSING ***")
    print(f"    missing by year: {dict(sorted(yrs.items()))}")
    print(
        f"    Re-run this script (it resumes) before evaluating. Metric scripts "
        f"will refuse this bank."
    )
    return False


def main():
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = max(ngpu, 1)

    stats = np.load(config.STATS_PATH, allow_pickle=True).item()
    schema = int(stats.get("schema", 1))
    if schema < 2:
        raise SystemExit("stats_allvars.npy is schema 1; re-run prepare_data.py")
    C = config.NUM_CHANNELS
    H, W = int(stats["H"]), int(stats["W"])
    n_seas5 = int(stats.get("n_members", 1))
    use_member = config.USE_MEMBER_BASELINE and n_seas5 > 1
    valid_sec = stats[f"valid_sec_{SPLIT}"]
    n_split = len(valid_sec)
    if n_split == 0:
        raise SystemExit(
            f"SPLIT='{SPLIT}' is empty — build that shard first "
            f"(prepare_data.py --reuse-stats --splits {SPLIT})."
        )
    pick = pick_dates(n_split)

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt = model_arch.resolve_ckpt(CKPT, CKPT_DIRS)

    per_date = (N_MEMBERS + 2) * C * H * W * 2
    est_gb = per_date * len(pick) / 1e9
    free_gb = shutil.disk_usage(OUT_DIR).free / 1e9
    print(
        f"Generating {len(pick)} dates x {N_MEMBERS} members, {SOLVER} "
        f"{INFERENCE_STEPS} steps, eta={ETA}"
    )
    print(f"  split={SPLIT}  out={OUT_DIR}  ckpt={ckpt}")
    print(
        f"  SEAS5 conditioning: {'per-member' if use_member else 'ensemble mean'} "
        f"({n_seas5} members available)"
    )
    print(f"  forward noise: {noise_mod.describe(config)}")
    print(
        f"  GPUs detected: {ngpu} (one worker each)"
        if ngpu
        else "  no CUDA — single process"
    )
    print(f"  estimated bank ~{est_gb:.1f} GB; free disk {free_gb:.1f} GB")
    if est_gb > 0.95 * free_gb:
        raise SystemExit(
            "Not enough free disk — free space or lower N_TIMES / N_MEMBERS."
        )

    np.savez(
        os.path.join(OUT_DIR, "manifest.npz"),
        schema=config.SCHEMA,
        indices=pick,
        valid_sec=valid_sec[pick],
        seed=SEED,
        n_members=N_MEMBERS,
        inference_steps=INFERENCE_STEPS,
        eta=ETA,
        solver=SOLVER,
        timestep_spacing=TIMESTEP_SPACING,
        seas5_members="per-member" if use_member else "ensmean",
        n_seas5_members=n_seas5,
        checkpoint=os.path.basename(ckpt),
        split=SPLIT,
        channels=np.array(list(stats["channels"])),
        truth_mean=stats["truth_mean"],
        truth_std=stats["truth_std"],
        res_mean=stats["res_mean"],
        res_std=stats["res_std"],
        base_mean=stats["base_mean"],
        base_std=stats["base_std"],
        lat_weights=stats.get("lat_weights", config.lat_weights()),
        recipe=(
            "schema 2: samples and truth are the FULL FIELD in truth "
            "normalisation -> field = samples*truth_std + truth_mean; "
            "baseline is the ensemble mean in base normalisation"
        ),
        **{k: np.asarray(v) for k, v in noise_mod.describe(config).items()},
    )

    run_metadata.write(
        os.path.join(OUT_DIR, "run_metadata.json"),
        dict(
            split=SPLIT,
            n_times=N_TIMES,
            n_dates=len(pick),
            n_members=N_MEMBERS,
            member_batch=MEMBER_BATCH,
            inference_steps=INFERENCE_STEPS,
            solver=SOLVER,
            eta=ETA,
            seed=SEED,
            timestep_spacing=TIMESTEP_SPACING,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=PREDICTION_TYPE,
            abar_min_x0=ABAR_MIN_X0,
            schema=config.SCHEMA,
            n_gpus=ngpu,
            seas5_members="per-member" if use_member else "ensmean",
            n_seas5_members=n_seas5,
            C=C,
            H=H,
            W=W,
            **noise_mod.describe(config),
        ),
        config,
        extra={"bank_dir": os.path.abspath(OUT_DIR), "checkpoint_path": ckpt},
    )

    if ngpu >= 2:
        mp.spawn(run_worker, args=(world_size, ckpt), nprocs=world_size, join=True)
    else:
        run_worker(0, world_size, ckpt)

    print()
    ok = check_complete(pick, valid_sec)
    print(f"{len(pick)} dates x {N_MEMBERS} members -> {OUT_DIR}/")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
