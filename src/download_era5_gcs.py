import os
import traceback
import concurrent.futures as cf
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp

import config
from download_era5 import int16_encoding

ARCO_URI = (
    "gs://gcp-public-data-arco-era5/ar/" "1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
)
ARCO_END_YEAR = 2021

START_YEAR = 1981
END_YEAR = ARCO_END_YEAR
NUM_WORKERS = 6

PL_VARS = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
}
SFC_VARS = {"sea_surface_temperature": "sst", "mean_sea_level_pressure": "msl"}

# The mirror downloads retain 1000 hPa; readers select the two model levels.
BATCH_LEVELS = [1000, 850, 500]

# Concurrent chunk reads within each monthly worker.
NUM_STREAMS = 4


def month_times(year, month):
    start = pd.Timestamp(year, month, 1)
    end = start + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=6)
    return pd.date_range(start, end, freq="6h")


def fetch_month(ds, var_map, times, with_levels, target):
    if os.path.exists(target):
        print(f"  skip (exists): {os.path.basename(target)}", flush=True)
        return
    lat_t, lon_t = config.grid_arrays()
    sub = ds[list(var_map)].sel(time=times)
    if with_levels:
        sub = sub.sel(level=BATCH_LEVELS)
    sub = sub.sel(latitude=lat_t, longitude=lon_t, method="nearest")
    sub = sub.assign_coords(latitude=lat_t, longitude=lon_t)
    rename = {**var_map, "time": "valid_time"}
    if with_levels:
        rename["level"] = "pressure_level"
    sub = sub.rename(rename)
    sub = sub.compute(num_workers=NUM_STREAMS)
    tmp = target + ".part"
    try:
        sub.to_netcdf(tmp, encoding=int16_encoding(sub))
        os.replace(tmp, target)
        print(f"  done:          {os.path.basename(target)}", flush=True)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


_DS = None

# Timeout per chunk request, in seconds.
REQUEST_TIMEOUT = 300

MONTH_ATTEMPTS = 2


def _init_worker():
    global _DS
    _DS = xr.open_zarr(
        ARCO_URI,
        chunks={},
        storage_options={
            "token": "anon",
            "timeout": REQUEST_TIMEOUT,
            "requests_timeout": REQUEST_TIMEOUT,
        },
    )


# Fetch one month (pl + sfc) in a worker process; returns a status line.
def _process_month(ym):
    year, month = ym
    tag = f"{year}{month:02d}"
    times = month_times(year, month)
    last_exc = None
    for attempt in range(1, MONTH_ATTEMPTS + 1):
        try:
            fetch_month(
                _DS,
                PL_VARS,
                times,
                True,
                os.path.join(config.ERA5_BATCH_DIR, f"era5_pl_{tag}.nc"),
            )
            fetch_month(
                _DS,
                SFC_VARS,
                times,
                False,
                os.path.join(config.ERA5_BATCH_DIR, f"era5_sfc_{tag}.nc"),
            )
            return f"ERA5 {tag}: ok"
        except Exception as exc:
            last_exc = exc

            print(
                f"  retry {tag} (attempt {attempt}): {type(exc).__name__}: {exc!r}",
                flush=True,
            )
            if attempt == MONTH_ATTEMPTS:
                traceback.print_exc()
    return f"ERA5 {tag}: FAILED {type(last_exc).__name__}: {last_exc!r}"


def main():
    if END_YEAR > ARCO_END_YEAR:
        raise SystemExit(
            f"ARCO 6-h store ends {ARCO_END_YEAR}-12-31; "
            f"use download_era5.py (CDS) for {ARCO_END_YEAR+1}+"
        )

    os.makedirs(config.ERA5_BATCH_DIR, exist_ok=True)
    months = [(y, m) for y in range(START_YEAR, END_YEAR + 1) for m in range(1, 13)]
    print(f"{len(months)} months, {NUM_WORKERS} workers, source {ARCO_URI}", flush=True)
    n_fail = 0

    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(
        max_workers=NUM_WORKERS, mp_context=ctx, initializer=_init_worker
    ) as ex:
        futures = {ex.submit(_process_month, ym): ym for ym in months}
        for fut in cf.as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                res = f"{futures[fut]} FAILED: {type(e).__name__}: {e}"
            n_fail += "FAILED" in res
            print(res, flush=True)
    print(
        f"All months processed ({n_fail} failures"
        f"{' - re-run to retry' if n_fail else ''})",
        flush=True,
    )
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
