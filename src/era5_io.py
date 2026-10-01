import glob
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

import config


# Standardise dim/coord names and longitude convention; select levels.
def _normalise(ds):
    rename = {}
    if "time" in ds.dims and "valid_time" not in ds.dims:
        rename["time"] = "valid_time"
    if "level" in ds.dims and "pressure_level" not in ds.dims:
        rename["level"] = "pressure_level"
    if rename:
        ds = ds.rename(rename)

    ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby("longitude")
    if "pressure_level" in ds.dims:
        ds = ds.sel(pressure_level=config.LEVELS)
    return ds


# Open all batches as one lazy, grid-normalised dataset (not yet interpolated).
def open_era5():

    # Subset levels per file before combining old and new download batches.
    pl = xr.open_mfdataset(
        sorted(glob.glob(f"{config.ERA5_BATCH_DIR}/era5_pl_*.nc")),
        combine="by_coords",
        preprocess=_normalise,
    )
    sfc = xr.open_mfdataset(
        sorted(glob.glob(f"{config.ERA5_BATCH_DIR}/era5_sfc_*.nc")),
        combine="by_coords",
        preprocess=_normalise,
    )
    ds = xr.merge([pl, sfc], join="inner")

    spacing = float(abs(np.diff(ds.latitude.values[:2])[0]))
    expected = (config.GRID_LAT[1] - config.GRID_LAT[0]) / (config.GRID_LAT[2] - 1)
    if not np.isclose(spacing, expected):
        raise RuntimeError(
            f"ERA5 batches are on a {spacing:g}-deg grid but config expects "
            f"{expected:g} deg - clear {config.ERA5_BATCH_DIR} and re-download"
        )
    return ds.sortby("valid_time")


# 6-hourly ERA5 timestamps as float seconds since 1970-01-01.
def time_axis_seconds():
    ds = open_era5()
    t = ds["valid_time"].values
    ds.close()
    epoch = np.datetime64("1970-01-01T00:00:00")
    return (t - epoch) / np.timedelta64(1, "s")


# Fill SST land NaNs from the nearest sea cell before interpolation.
def _fill_sst_nearest(sst):
    sst = np.array(sst, dtype=np.float32)
    for k in range(sst.shape[0]):
        nanmask = np.isnan(sst[k])
        if nanmask.any():
            idx = distance_transform_edt(
                nanmask, return_distances=False, return_indices=True
            )
            sst[k] = sst[k][tuple(idx)]
    return sst


# Return raw fields for timesteps [t0:t1] on the requested grid.
def raw_chunk_on_grid(ds, t0, t1, lat, lon):
    sub = ds.isel(valid_time=slice(t0, t1)).load()
    sub["sst"] = (sub["sst"].dims, _fill_sst_nearest(sub["sst"].values))
    sub = sub.interp(latitude=lat, longitude=lon, method="linear")
    raw = {}
    for v in config.BASE_PRESSURE_VARS:
        raw[v] = (
            sub[v]
            .transpose("valid_time", "pressure_level", "latitude", "longitude")
            .values.astype(np.float32)
        )
    for v in config.BASE_SURFACE_VARS:
        raw[v] = (
            sub[v]
            .transpose("valid_time", "latitude", "longitude")
            .values.astype(np.float32)
        )
    return raw
