import os
import sys
import glob
import json
import socket
import hashlib
import platform
import subprocess
from datetime import datetime

import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

VERSION_PACKAGES = (
    "torch",
    "diffusers",
    "numpy",
    "scipy",
    "pandas",
    "xarray",
    "netCDF4",
    "zarr",
    "gcsfs",
    "dask",
    "cdsapi",
    "matplotlib",
    "cartopy",
)

ENV_KEYS = (
    "ALLOW_CPU",
    "ALLOW_DEAD",
    "BATCH_SIZE",
    "CKPT",
    "EPOCHS",
    "ETA",
    "INFERENCE_STEPS",
    "MAX_DATES",
    "MEMBER_BATCH",
    "N_MEMBERS",
    "N_SAMPLES",
    "N_TIMES",
    "NUM_WORKERS",
    "ONLY_IDX",
    "ONLY_M",
    "OUT_DIR",
    "OUT_FOLDER",
    "PREP_WORKERS",
    "REPLOT",
    "RESUME",
    "SEED",
    "SMOKE_ARCH",
    "SOLVER",
    "SPLIT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)

GIT_TIMEOUT = 5
HASH_BLOCK = 1 << 20


# Convert arrays, scalars and containers to JSON-compatible values.
def _jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if callable(v) or isinstance(v, type(os)):
        return None
    return repr(v)


# Return installed distribution versions, or None for missing packages.
def _package_versions():
    from importlib import metadata

    out = {}
    for name in VERSION_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return out


# Collect PyTorch, CUDA and device information when available.
def _torch_info():
    info = {"torch": None, "cuda": None, "cudnn": None, "available": False, "gpus": []}
    try:
        import torch
    except Exception as exc:
        info["_error"] = f"torch not importable: {exc!r}"
        return info
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    try:
        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:

            info["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1e9, 1
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = bool(mps and mps.is_available())
    except Exception as exc:
        info["_error"] = repr(exc)
    return info


# Read the commit, branch and dirty status if code_dir is in a Git repository.
def _git_state(code_dir):

    def _git(*args):
        return subprocess.run(
            ("git",) + args,
            cwd=code_dir,
            timeout=GIT_TIMEOUT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
    except Exception:
        return {"commit": None, "note": "not a git working tree (or git unavailable)"}
    out = {"commit": commit}
    try:
        out["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
    except Exception:
        pass
    try:

        out["dirty"] = bool(_git("status", "--porcelain"))
    except Exception:
        pass
    return out


# Hex sha256 of one file, read in HASH_BLOCK chunks.
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(HASH_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


# Hash the Python files in code_dir, keyed by basename.
def code_hashes(code_dir=CODE_DIR):
    out = {}
    for path in sorted(glob.glob(os.path.join(code_dir, "*.py"))):
        try:
            out[os.path.basename(path)] = sha256_file(path)
        except Exception as exc:
            out[os.path.basename(path)] = f"unreadable: {exc!r}"
    return out


# Collect public uppercase configuration values as JSON-compatible data.
def config_dict(cfg):
    if cfg is None:
        return {}
    return {
        k: _jsonable(v)
        for k, v in sorted(vars(cfg).items())
        if k.isupper() and not k.startswith("_")
    }


# Collect run metadata, recording collector errors alongside successful fields.
def collect(hparams=None, cfg=None, when=None, code_dir=CODE_DIR, extra=None):
    when = when or datetime.now()
    rec = {
        "timestamp": when.isoformat(timespec="seconds"),
        "timestamp_unix": when.timestamp(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "code_dir": code_dir,
    }
    for key, fn in (
        ("versions", _package_versions),
        ("torch", _torch_info),
        ("git", lambda: _git_state(code_dir)),
        ("code_sha256", lambda: code_hashes(code_dir)),
    ):
        try:
            rec[key] = fn()
        except Exception as exc:
            rec[key] = {"_error": repr(exc)}
    rec["env"] = {k: os.environ[k] for k in ENV_KEYS if k in os.environ}
    rec["hparams"] = _jsonable(hparams or {})
    try:
        rec["config"] = config_dict(cfg)
    except Exception as exc:
        rec["config"] = {"_error": repr(exc)}
    if extra:
        rec.update(_jsonable(extra))
    return rec


# Write run metadata as indented JSON and return the record.
def write(path, hparams=None, cfg=None, when=None, code_dir=CODE_DIR, extra=None):
    rec = collect(hparams, cfg, when=when, code_dir=code_dir, extra=extra)
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:

            json.dump(rec, fh, indent=2, sort_keys=False, default=str)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"WARNING: could not write run metadata to {path}: {exc!r}", flush=True)
    return rec


if __name__ == "__main__":

    import config

    rec = collect(hparams={"example_hparam": 1}, cfg=config)
    text = json.dumps(rec, indent=2, default=str)
    print(text)
    print(
        f"\n{len(text)} bytes  |  {len(rec['code_sha256'])} .py files hashed in "
        f"{rec['code_dir']}  |  config keys: {len(rec['config'])}"
    )
