import os
import glob
import re

import xarray as xr

DATA_DIR = "seas5_data"
OUTPUT_DIR = "seas5_data"

CONCAT_DIM = "forecast_reference_time"
REGIMES = ["hc", "rt"]
GROUPS = ["pl", "sfc"]

COMBINE_REGIMES = False

COMP = dict(zlib=False)

_YYYYMM = re.compile(r"_(\d{6})\.nc$")


# All seas5_<group>_<regime>_<YYYYMM>.nc, sorted chronologically by YYYYMM.
def _sorted_files(group, regime):
    pattern = os.path.join(DATA_DIR, f"seas5_{group}_{regime}_*.nc")
    files = glob.glob(pattern)
    files = [f for f in files if _YYYYMM.search(f)]
    files.sort(key=lambda f: _YYYYMM.search(f).group(1))
    return files


# Open + concat every init-month for one (group, regime) along time, or None.
def _open_group(group, regime):
    files = _sorted_files(group, regime)
    if not files:
        print(f"    {group}: no files, skipping")
        return None
    print(
        f"    {group}: {len(files)} files "
        f"({_YYYYMM.search(files[0]).group(1)}..{_YYYYMM.search(files[-1]).group(1)})"
    )
    ds = xr.open_mfdataset(
        files,
        concat_dim=CONCAT_DIM,
        combine="nested",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=False,
    )

    ds = ds.sortby(CONCAT_DIM)
    return ds


# Build one consolidated dataset (pl+sfc vars) for a single regime, or None.
def merge_regime(regime):
    print(f"  regime '{regime}':")
    parts = [_open_group(g, regime) for g in GROUPS]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None

    ds = xr.merge(parts, compat="override", join="outer")
    n_members = ds.sizes.get("number")
    n_times = ds.sizes.get(CONCAT_DIM)
    print(
        f"    -> merged: {n_times} init-times, {n_members} members, "
        f"vars {list(ds.data_vars)}"
    )
    return ds


def _write(ds, path):
    encoding = {v: COMP for v in ds.data_vars}
    print(f"  writing {path} ...")
    ds.to_netcdf(path, encoding=encoding)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  done: {path}  ({size_mb:.1f} MB)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Merging SEAS5 files in '{DATA_DIR}'")

    merged = {}
    for regime in REGIMES:
        ds = merge_regime(regime)
        if ds is None:
            print(f"  regime '{regime}': nothing to merge")
            continue
        merged[regime] = ds
        out = os.path.join(OUTPUT_DIR, f"seas5_{regime}_merged.nc")
        _write(ds, out)

    if COMBINE_REGIMES and merged:
        print("Combining regimes into a single file...")
        regimes_present = [r for r in REGIMES if r in merged]
        combined = []
        for r in regimes_present:
            d = merged[r]

            d = d.assign_coords(regime=(CONCAT_DIM, [r] * d.sizes[CONCAT_DIM]))
            combined.append(d)

        ds_all = xr.concat(combined, dim=CONCAT_DIM, join="outer")
        ds_all = ds_all.sortby(CONCAT_DIM)
        out = os.path.join(OUTPUT_DIR, "seas5_all_merged.nc")
        _write(ds_all, out)

    print("Done")


if __name__ == "__main__":
    main()
