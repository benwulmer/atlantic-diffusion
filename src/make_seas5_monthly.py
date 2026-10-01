import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc

import config
from derived import assemble_17ch

EPOCH = pd.Timestamp("1970-01-01")

SOURCE_LEVELS = [850, 500]
SOURCE_LEVELS_PA = {1000: 100000.0, 850: 85000.0, 500: 50000.0}
SOURCE_CHANNELS = [
    f"{v}_{l}" for v in config.PRESSURE_VARS for l in SOURCE_LEVELS
] + config.SURFACE_VARS


# Read raw fields for the first configured members of one regime.
def load_regime_members(path):
    M = config.SEAS5_N_MEMBERS
    ds = xr.open_dataset(path)

    have = [int(x) for x in ds["pressure_level"].values]
    missing = [l for l in SOURCE_LEVELS if l not in have]
    assert not missing, f"{path}: pressure_level {have} is missing {missing}"
    ds = ds.sel(pressure_level=SOURCE_LEVELS)

    n_have = ds.sizes["number"]
    assert n_have >= M, (
        f"{path} has {n_have} members but config.SEAS5_N_MEMBERS={M}. Lower "
        f"SEAS5_N_MEMBERS (it must not exceed the hindcast's 25) or re-download."
    )
    ds = ds.isel(number=slice(0, M))

    raw = {}
    for v in config.BASE_PRESSURE_VARS:
        arr = ds[v].squeeze("forecastMonth", drop=True)
        raw[v] = arr.transpose(
            "forecast_reference_time",
            "number",
            "pressure_level",
            "latitude",
            "longitude",
        ).values.astype(np.float32)
    for v in config.BASE_SURFACE_VARS:
        arr = ds[v].squeeze("forecastMonth", drop=True)
        raw[v] = arr.transpose(
            "forecast_reference_time", "number", "latitude", "longitude"
        ).values.astype(np.float32)

    init_times = pd.to_datetime(ds["forecast_reference_time"].values)
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    ds.close()
    return raw, init_times, lat, lon


# Write the monthly field, with or without a member axis, to NetCDF.
def _write(path, field, t_sec, lat, lon, per_member):
    print(f"Writing {path}  shape {field.shape} ...")
    dst = nc.Dataset(path, "w")
    dst.createDimension("valid_time", len(t_sec))
    if per_member:
        dst.createDimension("member", field.shape[1])
    dst.createDimension("channel", len(SOURCE_CHANNELS))
    dst.createDimension("latitude", len(lat))
    dst.createDimension("longitude", len(lon))

    vt = dst.createVariable("valid_time", "f8", ("valid_time",))
    vt.units = "seconds since 1970-01-01"
    vt[:] = t_sec
    dst.createVariable("latitude", "f8", ("latitude",))[:] = lat
    dst.createVariable("longitude", "f8", ("longitude",))[:] = lon
    dims = (
        ("valid_time", "member", "channel", "latitude", "longitude")
        if per_member
        else ("valid_time", "channel", "latitude", "longitude")
    )
    if per_member:
        dst.createVariable("member", "i4", ("member",))[:] = np.arange(field.shape[1])
    fv = dst.createVariable("field", "f4", dims, zlib=True, complevel=4)
    fv[:] = field
    dst.channel_names = ",".join(SOURCE_CHANNELS)
    dst.n_members = int(field.shape[1]) if per_member else 1
    dst.close()


def main():
    parts, init_all = [], []
    lat = lon = None
    for path in (config.SEAS5_HC_MERGED, config.SEAS5_RT_MERGED):
        print(f"Loading {path} (first {config.SEAS5_N_MEMBERS} members) ...")
        raw, init_times, lat_i, lon_i = load_regime_members(path)
        if lat is None:
            lat, lon = lat_i, lon_i
        else:
            assert np.allclose(lat, lat_i) and np.allclose(
                lon, lon_i
            ), "hc / rt grids differ — regrid before merging"
        parts.append(raw)
        init_all.append(init_times)
        print(
            f"  {len(init_times)} init times "
            f"({init_times[0].date()} .. {init_times[-1].date()})"
        )

    raw = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    init_times = init_all[0].append(init_all[1])

    valid_times = init_times + pd.DateOffset(months=config.LEADTIME_MONTHS)
    valid_times = pd.DatetimeIndex(valid_times).to_period("M").to_timestamp()
    order = np.argsort(valid_times.values)
    valid_times = valid_times[order]
    for k in raw:
        raw[k] = raw[k][order]
    assert valid_times.is_unique, "duplicate valid months — check init/leadtime"
    print(
        f"Valid-time series: {valid_times[0].date()} .. {valid_times[-1].date()} "
        f"({len(valid_times)} months)"
    )

    print("Deriving rh + vorticity and assembling channels (per member) ...")
    field = assemble_17ch(
        raw,
        lat,
        lon,
        SOURCE_LEVELS,
        SOURCE_LEVELS_PA,
        config.PRESSURE_VARS,
        config.SURFACE_VARS,
        SOURCE_CHANNELS,
    )
    print(f"  field shape {field.shape} (T, M, C, H, W)  channels: {SOURCE_CHANNELS}")

    t_sec = np.array([(t - EPOCH).total_seconds() for t in valid_times], dtype="f8")
    _write(config.SEAS5_MONTHLY_MEMBERS, field, t_sec, lat, lon, per_member=True)

    _write(
        config.SEAS5_MONTHLY_17CH, field.mean(axis=1), t_sec, lat, lon, per_member=False
    )
    print("Done.")


if __name__ == "__main__":
    main()
