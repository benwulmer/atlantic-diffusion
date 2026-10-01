import os
import argparse
import threading
import concurrent.futures as cf
import cdsapi
import numpy as np
import xarray as xr

import config

START_YEAR = 2022
END_YEAR = 2025

LAT_NORTH = 60
LAT_SOUTH = 0
# Request the latitude band and crop longitude after download.
AREA = [LAT_NORTH, -180, LAT_SOUTH, 180]
GRID = ["0.25", "0.25"]

TIMES = ["00:00", "06:00", "12:00", "18:00"]
DAYS = [f"{d:02d}" for d in range(1, 32)]

BATCH_LEVELS = list(config.LEVELS)
PLEVELS = [str(l) for l in BATCH_LEVELS]

PL_DATASET = "reanalysis-era5-pressure-levels"
SFC_DATASET = "reanalysis-era5-single-levels"
PL_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
SFC_VARS = ["sea_surface_temperature", "mean_sea_level_pressure"]

NUM_PARALLEL = 6

_local = threading.local()
# Serialise NetCDF writes: HDF5 handles are not safely shared across threads.
_hdf5_lock = threading.Lock()


def _client():
    if not hasattr(_local, "c"):
        _local.c = cdsapi.Client()
    return _local.c


# Fit per-variable int16 packing with scale, offset and a NaN fill value.
def int16_encoding(ds):
    enc = {}
    for v in ds.data_vars:
        arr = ds[v].values
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
        span = max(vmax - vmin, 1e-9)
        enc[v] = {
            "dtype": "int16",
            "scale_factor": span / 64000.0,
            "add_offset": vmin + span / 2,
            "_FillValue": np.int16(-32767),
            "zlib": True,
            "complevel": 4,
        }
    return enc


# Merge raw part files, normalise longitudes to 0..360, crop to the grid.
def crop_to_box(src_paths, out_path):
    lat_t, lon_t = config.grid_arrays()
    ds = xr.open_mfdataset(src_paths, combine="by_coords")
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    ds = ds.drop_vars(["expver", "number"], errors="ignore")
    ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby("longitude")
    ds = ds.sel(latitude=lat_t, longitude=lon_t, method="nearest")

    ds = ds.assign_coords(latitude=lat_t, longitude=lon_t).sortby(tdim)
    ds = ds.compute()
    ds.to_netcdf(out_path, encoding=int16_encoding(ds))
    ds.close()


# Download and merge one monthly batch, retaining completed request parts.
def retrieve(dataset, requests, target):
    if os.path.exists(target):
        print(f"  skip (exists): {os.path.basename(target)}", flush=True)
        return True
    tmp_out = target + ".part"
    parts = []
    try:
        for k, req in enumerate(requests):
            p = f"{target}.raw{k}"
            if not os.path.exists(p):
                _client().retrieve(dataset, req, p + ".dl")
                os.replace(p + ".dl", p)
            parts.append(p)
        with _hdf5_lock:
            crop_to_box(parts, tmp_out)
        os.replace(tmp_out, target)
        for p in parts:
            os.remove(p)
        print(f"  done:          {os.path.basename(target)}", flush=True)
        return True
    except Exception as exc:
        for tmp in (f"{target}.raw{len(parts)}.dl", tmp_out):
            if os.path.exists(tmp):
                os.remove(tmp)
        print(
            f"  FAILED:        {os.path.basename(target)}\n                 {exc}",
            flush=True,
        )
        return False


# Download the pressure and surface fields for one month.
def _process_month(year, month, base):
    tag = f"{year}{month:02d}"
    print(f"ERA5 {tag}", flush=True)
    pl_parts = [
        {
            **base,
            "variable": [v],
            "pressure_level": PLEVELS,
            "day": DAYS,
            "year": str(year),
            "month": f"{month:02d}",
        }
        for v in PL_VARS
    ]
    ok_pl = retrieve(
        PL_DATASET, pl_parts, os.path.join(config.ERA5_BATCH_DIR, f"era5_pl_{tag}.nc")
    )
    ok_sfc = retrieve(
        SFC_DATASET,
        [{**base, "variable": SFC_VARS, "year": str(year), "month": f"{month:02d}"}],
        os.path.join(config.ERA5_BATCH_DIR, f"era5_sfc_{tag}.nc"),
    )
    return ok_pl and ok_sfc


def main():
    ap = argparse.ArgumentParser(description="Download 0.25-deg 6-h ERA5 batches")
    ap.add_argument("--start", type=int, default=START_YEAR)
    ap.add_argument("--end", type=int, default=END_YEAR)
    args = ap.parse_args()

    os.makedirs(config.ERA5_BATCH_DIR, exist_ok=True)
    base = {
        "product_type": "reanalysis",
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
        "grid": GRID,
        "day": DAYS,
        "time": TIMES,
    }
    months = [(y, m) for y in range(args.start, args.end + 1) for m in range(1, 13)]
    print(
        f"{len(months)} ERA5 months ({args.start}..{args.end}), {NUM_PARALLEL} in "
        f"parallel (already-downloaded files are skipped)",
        flush=True,
    )
    with cf.ThreadPoolExecutor(max_workers=NUM_PARALLEL) as ex:
        results = list(ex.map(lambda ym: _process_month(ym[0], ym[1], base), months))
    n_fail = results.count(False)
    print(
        f"ERA5 CDS pass complete: {len(months) - n_fail} ok, {n_fail} failed",
        flush=True,
    )
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
