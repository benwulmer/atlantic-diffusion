import os
import threading
import concurrent.futures as cf
import cdsapi

OUTPUT_DIR = "seas5_data"

START_YEAR = 1981
END_YEAR = 2025
HINDCAST_END_YEAR = 2016
INIT_MONTHS = [f"{m:02d}" for m in range(1, 13)]
LEADTIME_MONTHS = ["6"]

# North, west, south, east; includes a margin around the model grid.
AREA = [61, -91, -1, 1]
PRESSURE_LEVELS = ["850", "500"]
ORIGINATING_CENTRE = "ecmwf"
SYSTEM = "51"
PRODUCT_TYPE = ["monthly_mean"]

PRESSURE_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]

SINGLE_VARS = [
    "sea_surface_temperature",
    "mean_sea_level_pressure",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_PARALLEL = 20

# Each thread owns its CDS client and requests session.
_local = threading.local()


def _client():
    if not hasattr(_local, "c"):
        _local.c = cdsapi.Client()
    return _local.c


# Download one request atomically; return success, including existing files.
def retrieve(dataset, request, target):
    if os.path.exists(target):
        print(f"  skip (exists): {os.path.basename(target)}", flush=True)
        return True
    tmp = target + ".part"
    try:
        _client().retrieve(dataset, request, tmp)
        os.replace(tmp, target)
        print(f"  done:          {os.path.basename(target)}", flush=True)
        return True
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(
            f"  FAILED:        {os.path.basename(target)}\n                 {exc}",
            flush=True,
        )
        return False


def main():
    years = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
    base = {
        "originating_centre": ORIGINATING_CENTRE,
        "system": SYSTEM,
        "product_type": PRODUCT_TYPE,
        "leadtime_month": LEADTIME_MONTHS,
        "area": AREA,
        "data_format": "netcdf",
    }

    tasks = []
    for year in years:
        regime = "hc" if int(year) <= HINDCAST_END_YEAR else "rt"
        for month in INIT_MONTHS:
            tag = f"{regime}_{year}{month}"
            tasks.append(
                (
                    "seasonal-monthly-pressure-levels",
                    {
                        **base,
                        "variable": PRESSURE_VARS,
                        "pressure_level": PRESSURE_LEVELS,
                        "year": [year],
                        "month": [month],
                    },
                    os.path.join(OUTPUT_DIR, f"seas5_pl_{tag}.nc"),
                )
            )
            tasks.append(
                (
                    "seasonal-monthly-single-levels",
                    {**base, "variable": SINGLE_VARS, "year": [year], "month": [month]},
                    os.path.join(OUTPUT_DIR, f"seas5_sfc_{tag}.nc"),
                )
            )

    print(
        f"{len(tasks)} SEAS5 requests, {NUM_PARALLEL} in parallel "
        f"(already-downloaded files are skipped)",
        flush=True,
    )
    with cf.ThreadPoolExecutor(max_workers=NUM_PARALLEL) as ex:
        results = list(ex.map(lambda t: retrieve(*t), tasks))
    n_fail = results.count(False)
    print(
        f"SEAS5 download pass complete: {len(tasks) - n_fail} ok, {n_fail} failed",
        flush=True,
    )
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
