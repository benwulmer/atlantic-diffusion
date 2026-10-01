import os

# Limit BLAS threads before NumPy is imported in each worker.
for _v in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np

import config
import era5_io
import baseline_nodes
from derived import assemble_17ch

# Timesteps per I/O pass; statistics use a strided training sample.
CHUNK = 32
STATS_SAMPLES = 400
NUM_PREP_WORKERS = int(os.environ.get("PREP_WORKERS", min(os.cpu_count() or 8, 32)))


def years_of(valid_sec):
    return (np.datetime64("1970-01-01") + valid_sec.astype("timedelta64[s]")).astype(
        "datetime64[Y]"
    ).astype(int) + 1970


# Split by valid year using the fixed boundaries in config.py.
def make_split(valid_sec):
    yrs = years_of(valid_sec)
    val_lo = config.TEST_START_YEAR - config.VAL_YEARS
    test_idx = np.where(
        (yrs >= config.TEST_START_YEAR) & (yrs <= config.TEST_END_YEAR)
    )[0]
    val_idx = np.where((yrs >= val_lo) & (yrs < config.TEST_START_YEAR))[0]
    train_idx = np.where(yrs < val_lo)[0]

    def span(ix):
        return f"{int(yrs[ix].min())}..{int(yrs[ix].max())}" if len(ix) else "empty"

    print(f"  train: {len(train_idx)} ({span(train_idx)})")
    print(f"  val:   {len(val_idx)} ({span(val_idx)})")
    print(f"  test:  {len(test_idx)} ({span(test_idx)})")
    return {"train": train_idx, "val": val_idx, "test": test_idx}


# Assemble ERA5 timesteps [a:b] as (n, C, H, W) float32 fields.
def truth_chunk(ds, a, b, lat, lon):
    raw = era5_io.raw_chunk_on_grid(ds, a, b, lat, lon)
    return assemble_17ch(
        raw,
        lat,
        lon,
        config.LEVELS,
        config.PRESSURE_LEVELS_PA,
        config.PRESSURE_VARS,
        config.SURFACE_VARS,
        config.CHANNELS,
    )


# Sea = finite ERA5 SST (native land mask; sampled at a few timesteps).
def make_sea_mask(ds, era5_start, n, lat, lon):
    sample = [era5_start, era5_start + n // 2, era5_start + n - 1]
    sst = (
        ds["sst"]
        .isel(valid_time=sample)
        .sel(latitude=lat, longitude=lon, method="nearest")
        .values
    )
    mask = np.isfinite(sst).all(axis=0)
    np.save(config.SEA_MASK_PATH, {"mask": mask})
    print(f"  sea mask: {mask.sum()}/{mask.size} sea points -> {config.SEA_MASK_PATH}")
    return mask


# Estimate area-weighted channel means and standard deviations.
def channel_stats(
    ds,
    nodes_mean,
    nodes_mem,
    era5_start,
    train_idx,
    idx_all,
    w_all,
    lat,
    lon,
    n_members,
    lat_w,
):
    stride = max(1, len(train_idx) // STATS_SAMPLES)
    sel = train_idx[::stride]
    C = config.NUM_CHANNELS
    wt = lat_w[:, None]
    acc = np.zeros((6, C))

    def wmean(x):
        return (x * wt).mean(axis=(1, 2))

    for n_done, k in enumerate(sel, 1):
        truth = truth_chunk(ds, era5_start + k, era5_start + k + 1, lat, lon)[0]
        acc[0] += wmean(truth)
        acc[1] += wmean(truth.astype(np.float64) ** 2)
        base = baseline_nodes.eval_nodes(nodes_mean, idx_all[k], w_all[k])
        acc[2] += wmean(base)
        acc[3] += wmean(base.astype(np.float64) ** 2)
        if nodes_mem is None:
            res = truth - base
            acc[4] += wmean(res)
            acc[5] += wmean(res.astype(np.float64) ** 2)
        else:
            for m in range(n_members):
                bm = baseline_nodes.eval_nodes_member(
                    nodes_mem, m, idx_all[k], w_all[k]
                )
                res = truth - bm
                acc[4] += wmean(res) / n_members
                acc[5] += wmean(res.astype(np.float64) ** 2) / n_members
        if n_done % 25 == 0:
            print(f"  stats {n_done}/{len(sel)}", flush=True)
    acc /= len(sel)

    def finish(m1, m2):
        std = np.sqrt(np.maximum(m2 - m1**2, 0))
        std[std < 1e-8] = 1.0
        return m1.astype(np.float32), std.astype(np.float32)

    truth_mean, truth_std = finish(acc[0], acc[1])
    base_mean, base_std = finish(acc[2], acc[3])
    res_mean, res_std = finish(acc[4], acc[5])
    return (truth_mean, truth_std, res_mean, res_std, base_mean, base_std)


# Fill rows [r0:r1] of a prepared truth shard in a worker process.
def _write_block(args):
    (
        name,
        out,
        r0,
        r1,
        split_offset,
        era5_start,
        truth_mean,
        truth_std,
        lat,
        lon,
        marker_dir,
    ) = args
    marker = os.path.join(marker_dir, f"{name}.block_{r0}_{r1}.done")
    if os.path.exists(marker):
        return f"{name} rows {r0}:{r1} already done (marker present)"
    ds = era5_io.open_era5()
    mm = np.load(out, mmap_mode="r+")
    tm = truth_mean[:, None, None]
    ts = truth_std[:, None, None]
    for i in range(r0, r1, CHUNK):
        end = min(i + CHUNK, r1)
        aw, bw = split_offset + i, split_offset + end
        truth = truth_chunk(ds, era5_start + aw, era5_start + bw, lat, lon)
        for j in range(end - i):
            mm[i + j] = ((truth[j] - tm) / ts).astype(config.STORE_DTYPE)
        print(f"    [{name}] {end}/{r1} (block {r0}:{r1})", flush=True)
    mm.flush()
    del mm
    with open(marker, "w") as fh:
        fh.write(f"rows {r0}:{r1}\n")
    return f"{name} rows {r0}:{r1} done"


def write_split(name, idx, era5_start, lat, lon, stats):
    C, H, W = stats["C"], stats["H"], stats["W"]
    n = len(idx)
    assert np.all(np.diff(idx) == 1), "split indices must be contiguous"
    split_offset = int(idx[0])
    out = config.split_data_path(name)
    nbytes = n * C * H * W * np.dtype(config.STORE_DTYPE).itemsize
    print(f"  {out}: ({n}, {C}, {H}, {W}) {config.STORE_DTYPE}  {nbytes/1e9:.1f} GB")

    marker_dir = os.path.join(config.PREPPED_DIR, ".markers")
    os.makedirs(marker_dir, exist_ok=True)
    done_marker = os.path.join(marker_dir, f"{name}.complete")

    nworkers = max(1, min(NUM_PREP_WORKERS, n))
    bnd = np.linspace(0, n, nworkers + 1).astype(int)
    blocks = [
        (int(bnd[w]), int(bnd[w + 1])) for w in range(nworkers) if bnd[w] < bnd[w + 1]
    ]

    resuming = os.path.exists(out) and any(
        os.path.exists(os.path.join(marker_dir, f"{name}.block_{a}_{b}.done"))
        for a, b in blocks
    )
    if resuming:
        existing = np.load(out, mmap_mode="r")
        if existing.shape != (n, C, H, W):
            raise SystemExit(
                f"{out} exists with shape {existing.shape} but this run wants "
                f"{(n, C, H, W)}. Delete it and {marker_dir}/{name}.* to rebuild."
            )
        del existing
        print(f"  resuming: reusing {out} and skipping completed blocks")
    else:
        for a, b in blocks:
            m = os.path.join(marker_dir, f"{name}.block_{a}_{b}.done")
            if os.path.exists(m):
                os.remove(m)
        mm = np.lib.format.open_memmap(
            out, mode="w+", dtype=config.STORE_DTYPE, shape=(n, C, H, W)
        )
        del mm

    tasks = [
        (
            name,
            out,
            a,
            b,
            split_offset,
            era5_start,
            stats["truth_mean"],
            stats["truth_std"],
            lat,
            lon,
            marker_dir,
        )
        for a, b in blocks
    ]

    # Spawn avoids inheriting NetCDF handles and library threads.
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=len(tasks), mp_context=ctx) as ex:
        for res in ex.map(_write_block, tasks):
            print(f"  {res}", flush=True)

    missing = [
        (a, b)
        for a, b in blocks
        if not os.path.exists(os.path.join(marker_dir, f"{name}.block_{a}_{b}.done"))
    ]
    if missing:
        raise SystemExit(
            f"{name}: blocks {missing} did not complete - shard is "
            f"PARTIAL, do not train on it. Re-run to resume."
        )
    with open(done_marker, "w") as fh:
        fh.write(
            f"{n} rows, ({C},{H},{W}), {config.STORE_DTYPE}, schema {config.SCHEMA}\n"
        )
    print(f"  {name}: complete ({n} rows) -> {done_marker}")


# Normalise monthly nodes to fp16 in place-by-month (src may be a memmap).
def write_norm_nodes(path, src, base_mean, base_std, label):
    bm = base_mean[:, None, None]
    bs = base_std[:, None, None]
    nb = int(np.prod(src.shape)) * np.dtype(config.STORE_DTYPE).itemsize
    print(
        f"  {path}: {tuple(src.shape)} {config.STORE_DTYPE}  {nb/1e9:.1f} GB  ({label})"
    )
    nn = np.lib.format.open_memmap(
        path, mode="w+", dtype=config.STORE_DTYPE, shape=src.shape
    )
    for m in range(src.shape[0]):
        nn[m] = ((src[m].astype(np.float32) - bm) / bs).astype(config.STORE_DTYPE)
        if (m + 1) % 120 == 0:
            print(f"    {m+1}/{src.shape[0]}", flush=True)
    del nn


def main():
    ap = argparse.ArgumentParser(
        description="Prepare normalised ERA5 truth shards and SEAS5 baseline nodes."
    )
    ap.add_argument(
        "--reuse-stats",
        action="store_true",
        help="Reuse training normalisation statistics when adding a split.",
    )
    ap.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to write. Empty splits are skipped.",
    )
    ap.add_argument(
        "--skip-nodes-norm",
        action="store_true",
        help="Skip writing the normalised baseline node files.",
    )
    args = ap.parse_args()
    want = {s.strip() for s in args.splits.split(",") if s.strip()}

    os.makedirs(config.PREPPED_DIR, exist_ok=True)
    nodes_mean = np.load(config.BASELINE_NODES, mmap_mode="r")
    meta = np.load(config.BASELINE_NODES_META)
    node_sec = meta["node_sec"]
    lat, lon = meta["lat"], meta["lon"]
    H, W = len(lat), len(lon)
    C = config.NUM_CHANNELS
    lat_w = config.lat_weights()
    assert config.padded_shape() == (H, W), "grid must be a multiple of 16"

    n_members = int(meta["n_members"]) if "n_members" in meta.files else 1
    nodes_mem = None
    if config.USE_MEMBER_BASELINE:
        if n_members != config.SEAS5_N_MEMBERS:
            raise SystemExit(
                f"baseline nodes carry {n_members} members but config wants "
                f"{config.SEAS5_N_MEMBERS} - re-run make_seas5_monthly.py + "
                f"make_baseline.py, or set USE_MEMBER_BASELINE=False"
            )
        nodes_mem = np.load(config.BASELINE_NODES_MEM, mmap_mode="r")
        assert nodes_mem.shape == (nodes_mean.shape[0], n_members, C, H, W), (
            f"per-member nodes {nodes_mem.shape} inconsistent with "
            f"{(nodes_mean.shape[0], n_members, C, H, W)}"
        )
        print(f"Per-member baselines: {n_members} SEAS5 members")
    else:
        n_members = 1
        print("Per-member baselines DISABLED (schema-1 ensemble-mean conditioning)")

    ds = era5_io.open_era5()
    era5_sec = era5_io.time_axis_seconds()
    keep = np.where((era5_sec >= node_sec[0]) & (era5_sec <= node_sec[-1]))[0]
    assert np.all(np.diff(keep) == 1), "ERA5 axis has gaps inside the node span"
    era5_start = int(keep[0])
    valid_sec = era5_sec[keep]
    N = len(valid_sec)
    print(f"N={N} 6-h timesteps inside node span, grid ({H},{W}), C={C}")

    idx_all, w_all = baseline_nodes.catmull_rom_weights(valid_sec, node_sec)

    print("Building sea mask (ERA5-native)...")
    sea_mask = make_sea_mask(ds, era5_start, N, lat, lon)

    print("Building temporal split...")
    splits = make_split(valid_sec)

    if args.reuse_stats:
        if not os.path.exists(config.STATS_PATH):
            raise SystemExit(
                f"--reuse-stats but {config.STATS_PATH} not found "
                "(run phase 1 without --reuse-stats first)"
            )
        stats = np.load(config.STATS_PATH, allow_pickle=True).item()
        print(f"Reusing FROZEN stats from {config.STATS_PATH}")
    else:
        print("Estimating per-channel train stats (area-weighted)...")
        truth_mean, truth_std, res_mean, res_std, base_mean, base_std = channel_stats(
            ds,
            nodes_mean,
            nodes_mem,
            era5_start,
            splits["train"],
            idx_all,
            w_all,
            lat,
            lon,
            n_members,
            lat_w,
        )
        print(
            f"  {'channel':8s}{'truth mean':>12s}{'truth std':>11s}"
            f"{'res mean':>11s}{'res std':>11s}{'base mean':>12s}{'base std':>11s}"
        )
        for ci, nm in enumerate(config.CHANNELS):
            print(
                f"  {nm:8s}{truth_mean[ci]:12.4g}{truth_std[ci]:11.4g}"
                f"{res_mean[ci]:11.4g}{res_std[ci]:11.4g}"
                f"{base_mean[ci]:12.4g}{base_std[ci]:11.4g}"
            )
        stats = {
            "schema": config.SCHEMA,
            "channels": config.CHANNELS,
            "truth_mean": truth_mean,
            "truth_std": truth_std,
            "res_mean": res_mean,
            "res_std": res_std,
            "base_mean": base_mean,
            "base_std": base_std,
            "C": C,
            "H": H,
            "W": W,
            "H_pad": H,
            "W_pad": W,
            "n_members": n_members,
            "lat_weights": lat_w,
            "dtype": config.STORE_DTYPE,
            "layout": (
                "schema 2: shards = normalised TRUTH; baseline_m = Catmull-Rom "
                "over baseline_nodes_norm_mem.npy[.,m] with "
                "node_idx_/node_w_<split>; residual_m = truth - baseline_m"
            ),
        }

    stats["N"] = N
    for name, idx in splits.items():
        stats[f"node_idx_{name}"] = idx_all[idx]
        stats[f"node_w_{name}"] = w_all[idx]
        stats[f"valid_sec_{name}"] = valid_sec[idx]

    if args.skip_nodes_norm:
        print("Skipping normalised baseline nodes (--skip-nodes-norm)")
    else:
        print("Writing normalised baseline nodes (fp16)...")
        write_norm_nodes(
            config.NODES_NORM_PATH,
            nodes_mean,
            stats["base_mean"],
            stats["base_std"],
            "ensemble mean",
        )
        if nodes_mem is not None:
            write_norm_nodes(
                config.NODES_NORM_MEM_PATH,
                nodes_mem,
                stats["base_mean"],
                stats["base_std"],
                f"{n_members} SEAS5 members",
            )

    for name in ("train", "val", "test"):
        idx = splits[name]
        if name not in want:
            print(f"Skipping {name} shard (not in --splits)")
            continue
        if len(idx) == 0:
            print(f"Skipping {name} shard (no data for those years yet)")
            continue
        print(f"Writing {name} shard ({NUM_PREP_WORKERS} workers)...")
        write_split(name, idx, era5_start, lat, lon, stats)

    np.save(config.STATS_PATH, stats)
    np.save(config.SPLIT_PATH, splits)
    print(f"Saved {config.STATS_PATH}, {config.SPLIT_PATH}")
    print("\nDone. Needed on the GPU box for training:")
    for p in (
        config.split_data_path("train"),
        config.NODES_NORM_MEM_PATH if nodes_mem is not None else config.NODES_NORM_PATH,
        config.STATS_PATH,
        config.SEA_MASK_PATH,
    ):
        print(f"  {p}")
    print("  (+ src/ and requirements.txt)")
    print(
        f"\n{config.BASELINE_NODES_MEM} ({os.path.getsize(config.BASELINE_NODES_MEM)/1e9:.0f} GB) "
        f"is a scratch intermediate and can now be deleted."
        if nodes_mem is not None
        else ""
    )


if __name__ == "__main__":
    main()
