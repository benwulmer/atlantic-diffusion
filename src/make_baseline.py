import os
import numpy as np
import netCDF4 as nc
from scipy.ndimage import distance_transform_edt

import config
import baseline_nodes


# Return nearest valid-cell indices for a static SST NaN mask.
def nearest_sea_index_map(nan_mask):
    if not nan_mask.any():
        return None
    if nan_mask.all():
        raise RuntimeError(
            "SEAS5 SST is NaN everywhere - wrong variable or bad download"
        )
    return tuple(
        distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
    )


# Fill SST land cells in place using the nearest sea value.
def fill_sst_land(field, sst_idx):
    sst = field[..., sst_idx, :, :]
    nan_any = np.isnan(sst)
    if not nan_any.any():
        print("  SST land fill: no NaNs present (already filled?)")
        return 0

    flat = nan_any.reshape(-1, *sst.shape[-2:])
    static = flat[0]
    assert np.array_equal(flat, np.broadcast_to(static, flat.shape)), (
        "SEAS5 SST NaN mask varies by month/member - the land mask should be "
        "static; fill per slice instead"
    )
    idx = nearest_sea_index_map(static)
    filled = sst.reshape(-1, *sst.shape[-2:]).copy()
    for k in range(filled.shape[0]):
        filled[k] = filled[k][idx]
    field[..., sst_idx, :, :] = filled.reshape(sst.shape)
    print(
        f"  SST land fill: nearest-sea over {int(static.sum())}/{static.size} "
        f"land cells ({static.mean()*100:.1f}%)"
    )
    return int(static.sum())


# Indices/weights interpolating along one coordinate, edge-clamped.
def bilinear_1d(src, dst):
    src = np.asarray(src, dtype=np.float64)
    order = np.argsort(src)
    src_s = src[order]
    q = np.clip(np.asarray(dst, dtype=np.float64), src_s[0], src_s[-1])
    i1 = np.clip(np.searchsorted(src_s, q), 1, len(src_s) - 1)
    i0 = i1 - 1
    w = ((q - src_s[i0]) / (src_s[i1] - src_s[i0])).astype(np.float32)
    return order[i0], order[i1], w, q


# Reject grids that would require edge clamping outside the source box.
def _check_coverage(src, dst, q, name):
    n_clamped = int(np.sum(np.asarray(dst, dtype=np.float64) != q))
    if n_clamped:
        lo, hi = float(np.min(src)), float(np.max(src))
        raise SystemExit(
            f"{name}: {n_clamped} of {len(q)} target points lie outside the SEAS5 "
            f"range [{lo:g}, {hi:g}] and would EDGE-CLAMP to a constant.\n"
            f"  Working grid needs {float(np.min(dst)):g}..{float(np.max(dst)):g}. "
            f"Widen AREA in download_seas5.py (it must cover config.GRID_* plus ~1 deg) "
            f"and re-run the SEAS5 chain."
        )


def main():
    per_member = config.USE_MEMBER_BASELINE
    src_path = config.SEAS5_MONTHLY_MEMBERS if per_member else config.SEAS5_MONTHLY_17CH
    if not os.path.exists(src_path):
        raise SystemExit(f"{src_path} not found - run make_seas5_monthly.py first")

    ds = nc.Dataset(src_path)
    field = ds.variables["field"][:].astype(np.float32)
    month_start = ds.variables["valid_time"][:].astype(np.float64)
    src_lat = ds.variables["latitude"][:]
    src_lon = ds.variables["longitude"][:]
    src_names = ds.channel_names.split(",")
    ds.close()
    field = np.ma.getdata(field)

    node_sec = baseline_nodes.month_centroid_seconds(month_start)

    src_lon = np.asarray(src_lon, dtype=np.float64) % 360.0

    missing = [c for c in config.CHANNELS if c not in src_names]
    if missing:
        raise RuntimeError(
            f"monthly file lacks channels {missing} - rerun " "make_seas5_monthly.py"
        )
    pick = [src_names.index(c) for c in config.CHANNELS]
    field = field[..., pick, :, :]
    sst_idx = config.CHANNELS.index("sst")

    T_m = field.shape[0]
    M = field.shape[1] if per_member else 1
    C = config.NUM_CHANNELS
    print(
        f"Monthly nodes: {T_m}  members: {M}  channels: {C} "
        f"(of {len(src_names)} in file)  SEAS5 grid ({len(src_lat)},{len(src_lon)})"
    )
    print(
        f"Node times: {np.datetime64('1970-01-01') + node_sec[0].astype('timedelta64[s]')}"
        f" .. {np.datetime64('1970-01-01') + node_sec[-1].astype('timedelta64[s]')}"
    )

    fill_sst_land(field, sst_idx)

    lat, lon = config.grid_arrays()
    H, W = len(lat), len(lon)
    li0, li1, lw, qlat = bilinear_1d(src_lat, lat)
    ji0, ji1, jw, qlon = bilinear_1d(src_lon, lon)
    _check_coverage(src_lat, lat, qlat, "latitude")
    _check_coverage(src_lon, lon, qlon, "longitude")
    lw = lw[:, None]

    os.makedirs(os.path.dirname(config.BASELINE_NODES) or ".", exist_ok=True)

    # Bilinear (h,w) -> (H,W) on the trailing two axes of any-rank f.
    def interp_month(f):
        top = f[..., li0, :][..., :, ji0] * (1 - jw) + f[..., li0, :][..., :, ji1] * jw
        bot = f[..., li1, :][..., :, ji0] * (1 - jw) + f[..., li1, :][..., :, ji1] * jw
        return top * (1 - lw) + bot * lw

    mean_mm = np.lib.format.open_memmap(
        config.BASELINE_NODES, mode="w+", dtype="float32", shape=(T_m, C, H, W)
    )
    mem_mm = None
    if per_member:
        nb = T_m * M * C * H * W * 4
        print(
            f"Writing {config.BASELINE_NODES_MEM}  ({T_m}, {M}, {C}, {H}, {W}) "
            f"float32  {nb/1e9:.1f} GB"
        )
        mem_mm = np.lib.format.open_memmap(
            config.BASELINE_NODES_MEM,
            mode="w+",
            dtype="float32",
            shape=(T_m, M, C, H, W),
        )
    print(
        f"Writing {config.BASELINE_NODES}  ({T_m}, {C}, {H}, {W}) float32  "
        f"{T_m*C*H*W*4/1e9:.1f} GB"
    )

    for m in range(T_m):
        out = interp_month(field[m])
        if per_member:
            mem_mm[m] = out
            mean_mm[m] = out.mean(axis=0)
        else:
            mean_mm[m] = out
        if (m + 1) % 60 == 0:
            print(f"  {m+1}/{T_m}", flush=True)
    del mean_mm
    if mem_mm is not None:
        del mem_mm

    np.savez(
        config.BASELINE_NODES_META,
        node_sec=node_sec,
        lat=lat,
        lon=lon,
        n_members=M,
        per_member=per_member,
        schema=config.SCHEMA,
    )
    print(f"Saved {config.BASELINE_NODES_META}")
    print("Done.")


if __name__ == "__main__":
    main()
