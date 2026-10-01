import numpy as np
import torch

OFFSET_NOISE_WEIGHT = 0.02
NORMALISE_TOTAL_VARIANCE = False


# Broadcast a scalar or length-C sequence to a (c,) float64 array.
def _as_per_channel(v, c):
    a = np.atleast_1d(np.asarray(v, dtype=np.float64))
    if a.size == 1:
        return np.full(c, float(a[0]))
    if a.size != c:
        raise ValueError(
            f"OFFSET_NOISE_WEIGHT must be a scalar or {c} per-channel "
            f"values, got {a.size}"
        )
    return a


# Resolve (weight, normalise) from a config module, or the defaults.
def settings(cfg=None):
    if cfg is None:
        return OFFSET_NOISE_WEIGHT, NORMALISE_TOTAL_VARIANCE
    return (
        getattr(cfg, "OFFSET_NOISE_WEIGHT", OFFSET_NOISE_WEIGHT),
        bool(getattr(cfg, "NORMALISE_TOTAL_VARIANCE", NORMALISE_TOTAL_VARIANCE)),
    )


# Return the active noise settings for run metadata.
def describe(cfg=None):
    w, norm = settings(cfg)
    return {
        "noise_kind": "offset",
        "offset_noise_weight": np.atleast_1d(np.asarray(w, dtype=float)).tolist(),
        "normalise_total_variance": norm,
    }


# Draw offset noise with shape (B, C, H, W).
def noise_like(shape, device, generator=None, cfg=None, dtype=torch.float32):
    b, c, _, _ = shape
    raw_weight, normalise = settings(cfg)
    weight = _as_per_channel(raw_weight, c)

    out = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
    if np.any(weight > 0):
        wt = torch.as_tensor(weight, device=device, dtype=torch.float32).view(
            1, c, 1, 1
        )

        z = torch.randn(
            (b, c, 1, 1), device=device, generator=generator, dtype=torch.float32
        )
        out = out + wt * z
        if normalise:
            out = out / torch.sqrt(1.0 + wt**2)
    return out.to(dtype)


# Domain-mean noise SD per channel before variance renormalisation.
def domain_mean_sd(cfg=None, h=240, w=304, n_channels=None):
    import config

    c = config.NUM_CHANNELS if n_channels is None else n_channels
    weight = _as_per_channel(settings(cfg)[0], c)
    return np.sqrt(1.0 / (h * w) + weight**2)


# Verify the noise law, the per-channel path, and reproducibility.
def _selftest():
    import config

    C = config.NUM_CHANNELS
    h, w_ = 240, 304

    print("=== noise law ===")
    g = torch.Generator().manual_seed(1)
    x = noise_like((512, C, 64, 80), "cpu", generator=g, cfg=config)
    weight = _as_per_channel(settings(config)[0], C)
    exp_var = 1.0 + weight**2 if not settings(config)[1] else np.ones(C)
    got_var = x.var(dim=(0, 2, 3), unbiased=True).numpy()
    print(
        f"  per-pixel variance: expected 1+w^2 = {exp_var[0]:.6f}, "
        f"got {got_var.mean():.6f}"
    )
    assert abs(got_var.mean() - exp_var.mean()) < 0.02, "variance is wrong"

    dm = x.mean(dim=(2, 3))
    got_dm = dm.std(dim=0, unbiased=True).numpy()
    want_dm = np.sqrt(1.0 / (64 * 80) + weight**2)
    print(
        f"  domain-mean sd:     expected {want_dm[0]:.5f}, got {got_dm.mean():.5f} "
        f"(white would be {np.sqrt(1.0/(64*80)):.5f})"
    )
    assert (
        abs(got_dm.mean() - want_dm.mean()) < 0.15 * want_dm.mean()
    ), "the offset is not reaching the domain mean"

    print("\n=== per-channel weights are honoured ===")

    class _Cfg:
        OFFSET_NOISE_WEIGHT = config.OFFSET_MATCHED_WEIGHT
        NORMALISE_TOTAL_VARIANCE = False

    g = torch.Generator().manual_seed(2)
    y = noise_like((512, C, 64, 80), "cpu", generator=g, cfg=_Cfg)
    got = y.mean(dim=(2, 3)).std(dim=0, unbiased=True).numpy()
    want = np.sqrt(1.0 / (64 * 80) + np.asarray(config.OFFSET_MATCHED_WEIGHT) ** 2)
    print(f"  {'channel':9s}{'target':>9s}{'measured':>10s}")
    for ci in range(C):
        print(f"  {config.CHANNELS[ci]:9s}{want[ci]:9.4f}{got[ci]:10.4f}")
    rel = np.abs(got - want) / want
    assert rel.max() < 0.15, f"per-channel offset off by {rel.max()*100:.0f}%"

    print("\n=== reproducibility (train/sample must agree) ===")
    a = noise_like(
        (2, C, 64, 80), "cpu", generator=torch.Generator().manual_seed(7), cfg=config
    )
    b = noise_like(
        (2, C, 64, 80), "cpu", generator=torch.Generator().manual_seed(7), cfg=config
    )
    assert torch.equal(a, b), "same seed must give identical noise"
    print("  same seed -> identical noise: OK")

    print(f"\ndomain-mean sd at the working grid ({h}x{w_}):")
    sd = domain_mean_sd(config, h, w_)
    print(f"  {sd[0]:.5f} (white: {np.sqrt(1.0/(h*w_)):.5f})")
    print(f"\nactive config: {describe(config)}")


if __name__ == "__main__":
    _selftest()
