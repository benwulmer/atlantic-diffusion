import numpy as np

EARTH_RADIUS = 6.371e6


# Relative humidity (%) from q (kg/kg), t (K) and p (Pa), using Bolton (1980).
def relative_humidity(q, t, p_pa):
    q = np.asarray(q, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    e = q * p_pa / (0.622 + 0.378 * q)
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    rh = 100.0 * e / es

    return np.clip(rh, 0.0, 120.0).astype(np.float32)


# Spherical relative vorticity (s^-1) on latitude/longitude coordinates.
def relative_vorticity(u, v, lat, lon):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lat_rad = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(lon, dtype=np.float64))

    dv_dlon = np.gradient(v, lon_rad, axis=-1)
    du_dlat = np.gradient(u, lat_rad, axis=-2)

    coslat = np.cos(lat_rad)
    coslat = np.clip(coslat, 1e-6, None)
    tanlat = np.tan(lat_rad)
    coslat = coslat[:, None]
    tanlat = tanlat[:, None]

    dvdx = dv_dlon / (EARTH_RADIUS * coslat)
    dudy = du_dlat / EARTH_RADIUS
    # Spherical metric term from meridian convergence.
    metric = u * tanlat / EARTH_RADIUS
    zeta = dvdx - dudy + metric
    return zeta.astype(np.float32)


# Assemble (..., C, H, W) fields in the requested channel order.
def assemble_17ch(
    raw, lat, lon, levels, pressure_levels_pa, pressure_vars, surface_vars, channels
):
    lead = raw["t"].shape[:-3]
    H, W = raw["t"].shape[-2:]
    out = np.empty(lead + (len(channels), H, W), dtype=np.float32)

    derived_by_name = {}
    for li, lvl in enumerate(levels):
        p_pa = pressure_levels_pa[lvl]
        t_l = raw["t"][..., li, :, :]
        u_l = raw["u"][..., li, :, :]
        v_l = raw["v"][..., li, :, :]
        q_l = raw["q"][..., li, :, :]
        derived_by_name[f"rh_{lvl}"] = relative_humidity(q_l, t_l, p_pa)
        derived_by_name[f"vo_{lvl}"] = relative_vorticity(u_l, v_l, lat, lon)
        derived_by_name[f"u_{lvl}"] = u_l.astype(np.float32)
        derived_by_name[f"v_{lvl}"] = v_l.astype(np.float32)
        derived_by_name[f"t_{lvl}"] = t_l.astype(np.float32)

    for sv in surface_vars:
        derived_by_name[sv] = np.asarray(raw[sv], dtype=np.float32)

    for ci, name in enumerate(channels):
        out[..., ci, :, :] = derived_by_name[name]
    return out
