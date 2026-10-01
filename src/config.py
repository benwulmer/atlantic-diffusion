import numpy as np

# Channel order is shared by shards, checkpoints and output banks.
PRESSURE_VARS = ["rh", "vo", "u", "v", "t"]
LEVELS = [850, 500]
SURFACE_VARS = ["sst", "msl"]

CHANNELS = [f"{v}_{lvl}" for v in PRESSURE_VARS for lvl in LEVELS] + SURFACE_VARS
NUM_CHANNELS = len(CHANNELS)

_VAR_FULL = {
    "rh": "Relative Humidity",
    "vo": "Relative Vorticity",
    "u": "U Component of Wind",
    "v": "V Component of Wind",
    "t": "Temperature",
    "sst": "Sea Surface Temperature",
    "msl": "Mean Sea Level Pressure",
}


# Full label for a channel code, e.g. 'rh_850' -> 'Relative Humidity at 850 hPa'.
def pretty_name(name):
    if "_" in name:
        var, lev = name.split("_")
        return f"{_VAR_FULL.get(var, var)} at {lev} hPa"
    return _VAR_FULL.get(name, name)


BASE_PRESSURE_VARS = ["t", "u", "v", "q"]
BASE_SURFACE_VARS = ["sst", "msl"]

# Native 0.25-degree points; longitude uses the 0..360 convention.
GRID_LAT = (0.0, 59.75, 240)
GRID_LON = (275.0, 350.75, 304)
PAD_MULTIPLE = 16

# forecastMonth=6 is the sixth forecast month: init month + 5.
LEADTIME_MONTHS = 5
PRESSURE_LEVELS_PA = {850: 85000.0, 500: 50000.0}

# Use the first 25 members in both hindcast and real-time forecasts.
SEAS5_N_MEMBERS = 25
USE_MEMBER_BASELINE = True

OFFSET_NOISE_WEIGHT = 0.02
NORMALISE_TOTAL_VARIANCE = False

# Diagnostic alternative weights; the model uses the scalar 0.02 above.
OFFSET_MATCHED_WEIGHT = [
    0.147,
    0.128,
    0.033,
    0.039,
    0.107,
    0.122,
    0.100,
    0.094,
    0.205,
    0.253,
    0.248,
    0.337,
]

RESIDUAL_SLOPE_P = [
    3.48,
    4.05,
    2.71,
    3.15,
    4.28,
    4.70,
    4.24,
    4.67,
    4.10,
    4.66,
    2.58,
    5.02,
]

# Offsets are in six-hourly steps and clipped to +/-3 steps.
TEMPORAL_SMEAR_SIGMA = 1.5

TEMPORAL_SMEAR_MAX = 3

# Split by valid year: train <=2018, validation 2019-2021, test 2022-2024.
TEST_START_YEAR = 2022
TEST_END_YEAR = 2024
VAL_YEARS = 3

# Paths are relative to the repository root. Historical data filenames are retained.
SEAS5_DIR = "seas5_data"
SEAS5_HC_MERGED = "seas5_data/seas5_hc_merged.nc"
SEAS5_RT_MERGED = "seas5_data/seas5_rt_merged.nc"
SEAS5_MONTHLY_17CH = "seas5_data/seas5_monthly_17ch.nc"
SEAS5_MONTHLY_MEMBERS = "seas5_data/seas5_monthly_members.nc"

ERA5_DIR = "era5_data"
ERA5_BATCH_DIR = "era5_data/batches"

BASELINE_NODES = "data/baseline_nodes.npy"
BASELINE_NODES_MEM = "data/baseline_nodes_mem.npy"
BASELINE_NODES_META = "data/baseline_nodes_meta.npz"
NODES_NORM_PATH = "prepped/baseline_nodes_norm.npy"
NODES_NORM_MEM_PATH = "prepped/baseline_nodes_norm_mem.npy"

PREPPED_DIR = "prepped"
STORE_DTYPE = "float16"
SCHEMA = 2
SPLIT_PATH = "split_allvars.npy"
STATS_PATH = "stats_allvars.npy"
SEA_MASK_PATH = "sea_mask.npy"


# Return (lat, lon) 1-D float64 coordinate arrays of the working grid.
def grid_arrays():
    lat = np.linspace(GRID_LAT[0], GRID_LAT[1], GRID_LAT[2])
    lon = np.linspace(GRID_LON[0], GRID_LON[1], GRID_LON[2])
    return lat, lon


# Cosine-latitude area weights, normalised to mean one.
def lat_weights():
    lat, _ = grid_arrays()
    w = np.cos(np.deg2rad(lat))
    return (w / w.mean()).astype(np.float32)


# Path to the float16 normalised-truth shard for a split.
def split_data_path(name):
    import os

    return os.path.join(PREPPED_DIR, f"data_{name}.npy")


# Return (H_pad, W_pad) for the configured grid.
def padded_shape():
    h, w = GRID_LAT[2], GRID_LON[2]
    pad_h = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
    pad_w = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
    return h + pad_h, w + pad_w
