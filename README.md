# Atlantic residual diffusion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23149630.svg)](https://doi.org/10.5281/zenodo.23149630)

Conditional diffusion downscaling of ECMWF SEAS5 seasonal forecasts over the North
Atlantic. The model generates weather fields by adding a learned residual to an
individual SEAS5 ensemble member, using ERA5 as the training target.

## Citation

For work using version 1.0.0, please cite the archived release:

> Ulmer, B. (2026). *Atlantic residual diffusion* (Version 1.0.0) [Software].
> Zenodo. https://doi.org/10.5281/zenodo.23149631

The [all-versions DOI](https://doi.org/10.5281/zenodo.23149630) identifies the
project across releases. Citation metadata is also available in
[CITATION.cff](CITATION.cff).

## Setup

Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use Python 3.10 or newer. Training was written for Linux with an NVIDIA GPU and a
CUDA-enabled PyTorch installation.

## Download the data

Set up a Copernicus Climate Data Store account, accept the terms for the datasets
below, and put your API configuration in `~/.cdsapirc`, following the
[CDS setup instructions](https://cds.climate.copernicus.eu/how-to-api):

```yaml
url: https://cds.climate.copernicus.eu/api
key: <your-personal-access-token>
```

| Input | CDS dataset |
| --- | --- |
| SEAS5 pressure fields | `seasonal-monthly-pressure-levels` |
| SEAS5 surface fields | `seasonal-monthly-single-levels` |
| ERA5 pressure fields | `reanalysis-era5-pressure-levels` |
| ERA5 surface fields | `reanalysis-era5-single-levels` |

```bash
python src/download_seas5.py
python src/download_era5.py --start 1981 --end 2024
```

SEAS5 uses system 51, forecast month 6, and initialisations from 1981 through 2025.
Forecast month 6 is valid five calendar months after initialisation. The year
range, region and download concurrency are set near the top of `download_seas5.py`.

For ERA5 through 2021, the public ARCO mirror is an alternative:

```bash
python src/download_era5_gcs.py
python src/download_era5.py --start 2022 --end 2024
```

Both ERA5 routes write `era5_data/batches/`. ARCO needs no Google credentials, but
its global chunks cause much larger transfers than the regional CDS requests.

Just as a heasd up, full data preparation needs hundreds of GB of disk space.

## Prepare the inputs

```bash
python src/merge_seas5.py
python src/make_seas5_monthly.py
python src/make_baseline.py
PREP_WORKERS=4 python src/prepare_data.py
```

Splits use: training through 2018, validation in 2019–2021,
and test in 2022–2024. Normalisation statistics are estimated from training data.


## Train

We used four GPUs with eight examples per GPU:

```bash
OUT_FOLDER=training BATCH_SIZE=8 EPOCHS=60 NUM_WORKERS=4 SEED=0 \
  PREDICTION_TYPE=v_prediction HIGH_T_FRACTION=0 \
  torchrun --standalone --nproc_per_node=4 src/train.py
```

The model uses 1,000 cosine diffusion steps and a five-resolution U-Net with
approximately 161 million parameters. Its loss is 0.9 times the min-SNR-weighted
**L1** error plus 0.1 times a per-channel histogram KL term. 

Periodic checkpoints, resume state and run metadata go in `training/`. The final
raw and EMA weights go in `checkpoints/training/`. Use
`checkpoints/training/allvars_ema_final.pt` for sampling.


## Sampling

For a sampling 500 dates with twenty members per date:

```bash
CKPT=checkpoints/training/allvars_ema_final.pt \
  SPLIT=test N_TIMES=500 N_MEMBERS=20 MEMBER_BATCH=5 \
  SOLVER=ddim INFERENCE_STEPS=500 ETA=0.5 SEED=0 \
  PREDICTION_TYPE=v_prediction ABAR_MIN_X0=0 \
  OUT_DIR=outputs/test_500x20 python src/sample.py
```

Use `SPLIT=val` and a separate output directory for validation. Each run writes
`manifest.npz`, `run_metadata.json`, and one `sample_XXXXXX.npz` per selected date.


## Output layout

| Path | Contents |
| --- | --- |
| `seas5_data/` | Downloaded forecasts and merged monthly fields |
| `era5_data/batches/` | Monthly pressure and surface reanalysis batches |
| `data/` | Physical monthly baseline nodes and grid/time metadata |
| `prepped/` | Float16 normalised truth shards and baseline nodes |
| `stats_allvars.npy` | Channel statistics, split timestamps and interpolation weights |
| `split_allvars.npy` | Train, validation and test indices |
| `sea_mask.npy` | ERA5 sea mask, with one for sea and zero for land |
| `training/`, `checkpoints/training/` | Training state and model weights |
| `outputs/` | Generated sample banks |

In a sample file, `samples` has shape `(members, 12, 240, 304)`. It stores the
**full generated field in truth normalisation**. Optional `truth` has shape
`(12, 240, 304)` and uses the same normalisation. `baseline` is the SEAS5 ensemble
mean in baseline normalisation; `member_ids` identifies the conditioning members.

To recover physical units:

```python
import numpy as np

manifest = np.load("outputs/test_500x20/manifest.npz")
index = int(manifest["indices"][0])
sample = np.load(f"outputs/test_500x20/sample_{index:06d}.npz")
mean = manifest["truth_mean"][None, :, None, None]
std = manifest["truth_std"][None, :, None, None]
fields = sample["samples"].astype(np.float32) * std + mean
```

Channel order is `rh_850, rh_500, vo_850, vo_500, u_850, u_500, v_850, v_500,
t_850, t_500, sst, msl`. 

## Source files

| Stage | Files in `src/` |
| --- | --- |
| Acquisition | `download_seas5.py`, `download_era5.py`, `download_era5_gcs.py` |
| Preparation | `merge_seas5.py`, `make_seas5_monthly.py`, `make_baseline.py`, `prepare_data.py` |
| Training and sampling | `train.py`, `sample.py`, `model_arch.py`, `noise.py` |
| Shared utilities | `config.py`, `baseline_nodes.py`, `derived.py`, `era5_io.py`, `run_metadata.py` |

## Data licences

The code is distributed under the MIT licence in [LICENSE](LICENSE).
ERA5 and SEAS5 are Copernicus Climate Change Service data supplied through ECMWF;
their dataset terms apply separately. The ARCO mirror contains reformatted ERA5
data. Users must accept the applicable dataset terms and acknowledge the data
providers in work using these inputs. ECMWF and the Copernicus Climate Change
Service are not responsible for use of the data or generated model outputs.
