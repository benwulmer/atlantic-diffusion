import numpy as np


# Return (N, 4) node indices and weights for query times in seconds.
def catmull_rom_weights(query_sec, node_sec):
    node_sec = np.asarray(node_sec, dtype=np.float64)
    q = np.atleast_1d(np.asarray(query_sec, dtype=np.float64))
    if q.min() < node_sec[0] or q.max() > node_sec[-1]:
        raise ValueError("query times outside the node span")

    i = np.searchsorted(node_sec, q, side="right") - 1
    i = np.clip(i, 0, len(node_sec) - 2)
    im1 = np.clip(i - 1, 0, len(node_sec) - 1)
    ip2 = np.clip(i + 2, 0, len(node_sec) - 1)

    h = node_sec[i + 1] - node_sec[i]
    s = (q - node_sec[i]) / h
    # Cubic Hermite basis on the local time interval.
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2

    dm = node_sec[i + 1] - node_sec[im1]
    dp = node_sec[ip2] - node_sec[i]

    w = np.empty((len(q), 4), dtype=np.float32)
    w[:, 0] = -h10 * h / dm
    w[:, 1] = h00 - h11 * h / dp
    w[:, 2] = h01 + h10 * h / dm
    w[:, 3] = h11 * h / dp
    idx = np.stack([im1, i, i + 1, ip2], axis=1).astype(np.int32)
    return idx, w


# Interpolate one (C, H, W) field from a (T, C, H, W) node array.
def eval_nodes(nodes, idx, w):
    out = nodes[int(idx[0])].astype(np.float32) * w[0]
    for j in range(1, 4):
        out += nodes[int(idx[j])].astype(np.float32) * w[j]
    return out


# Interpolate one member from a (T, M, C, H, W) node array.
def eval_nodes_member(nodes_mem, member, idx, w):
    m = int(member)
    out = nodes_mem[int(idx[0]), m].astype(np.float32) * w[0]
    for j in range(1, 4):
        out += nodes_mem[int(idx[j]), m].astype(np.float32) * w[j]
    return out


# Return each month's midpoint in seconds since 1970-01-01.
def month_centroid_seconds(month_start_sec):
    t = np.asarray(month_start_sec, dtype=np.float64)

    starts = np.datetime64("1970-01-01") + t.astype("timedelta64[s]")
    months = starts.astype("datetime64[M]")
    nxt = (months + 1).astype("datetime64[s]").astype(np.float64)
    length = nxt - months.astype("datetime64[s]").astype(np.float64)
    return months.astype("datetime64[s]").astype(np.float64) + 0.5 * length
