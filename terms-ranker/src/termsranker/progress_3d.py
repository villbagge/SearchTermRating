from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple

def _extract_arrays(terms) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratings = np.array([float(getattr(t, "rating", 0.0)) for t in terms], dtype=float)
    games   = np.array([float(getattr(t, "games", 0.0))  for t in terms], dtype=float)
    sigma   = np.array([float(getattr(t, "sigma", 0.0))  for t in terms], dtype=float)
    ratings[~np.isfinite(ratings)] = np.nan
    games[~np.isfinite(games)] = np.nan
    sigma[~np.isfinite(sigma)] = np.nan
    return ratings, games, sigma

def _nice_linspace(vmin: float, vmax: float, nbins: int) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 1000.0, 2000.0
    return np.linspace(vmin, vmax, nbins+1)

def _log_bins(min_positive: float, max_value: float, nbins: int) -> np.ndarray:
    lo = max(1.0, float(min_positive))
    hi = max(lo * 1.0001, float(max_value))
    edges = np.geomspace(lo, hi, nbins+1)
    edges = np.concatenate(([0.0], edges))  # bucket for 0
    return edges

def _binned_stat_2d(x, y, z, x_edges, y_edges, agg: str = "median"):
    x_idx = np.digitize(x, x_edges) - 1
    y_idx = np.digitize(y, y_edges) - 1
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    Z = np.full((ny, nx), np.nan, dtype=float)
    N = np.zeros((ny, nx), dtype=int)
    for i in range(nx):
        for j in range(ny):
            mask = (x_idx == i) & (y_idx == j) & np.isfinite(z)
            if not np.any(mask):
                continue
            vals = z[mask]
            N[j, i] = vals.size
            Z[j, i] = float(np.median(vals)) if agg == "median" else float(np.mean(vals))
    return Z, N

def _iterative_nan_fill(Z: np.ndarray, max_passes: int = 6) -> np.ndarray:
    """Fill NaNs by averaging available 4-neighbors iteratively (no SciPy)."""
    Zf = Z.copy()
    for _ in range(max_passes):
        if not np.any(np.isnan(Zf)): break
        Zpad = np.pad(Zf, 1, mode="edge")
        valid = ~np.isnan(Zpad)
        # neighbor sums
        s = np.zeros_like(Zpad, dtype=float)
        c = np.zeros_like(Zpad, dtype=float)
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            nb = np.roll(np.roll(Zpad, dy, axis=0), dx, axis=1)
            vmask = np.roll(np.roll(valid, dy, axis=0), dx, axis=1)
            nb = np.where(vmask, nb, 0.0)
            s += nb
            c += vmask.astype(float)
        avg = np.where(c>0, s/np.maximum(c,1.0), np.nan)
        inner = avg[1:-1,1:-1]
        Zf = np.where(np.isnan(Zf), inner, Zf)
    return Zf

def _gaussian_smooth_nan(Z: np.ndarray, sigma_xy=(1.0,1.0), passes: int = 1) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return Z
    Zs = Z.copy()
    valid = np.isfinite(Zs).astype(float)
    Zs[np.isnan(Zs)] = 0.0
    for _ in range(passes):
        num = gaussian_filter(Zs, sigma=sigma_xy, mode="nearest")
        den = gaussian_filter(valid, sigma=sigma_xy, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            Zs = num / np.maximum(den, 1e-9)
            Zs[den < 1e-6] = np.nan
    return Zs

def show_progress_3d(terms: Iterable):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    ratings, games, sigma = _extract_arrays(terms)
    mask = np.isfinite(ratings) & np.isfinite(games) & np.isfinite(sigma)
    ratings, games, sigma = ratings[mask], games[mask], sigma[mask]
    if ratings.size == 0:
        raise RuntimeError("No valid data for 3D progress view.")

    # ----- Adaptive bins (avoid sparse grids)
    n = ratings.size
    nbins_x = int(np.clip(round(np.sqrt(n)/2), 8, 20))   # 8..20 based on data volume
    nbins_y = int(np.clip(round(np.sqrt(n)/3), 6, 16))   # 6..16
    x_edges = _nice_linspace(np.nanmin(ratings), np.nanmax(ratings), nbins_x)
    pos_games = games[games > 0]
    min_pos = float(np.nanmin(pos_games)) if pos_games.size else 1.0
    max_g = float(np.nanmax(games)) if np.any(np.isfinite(games)) else 1.0
    y_edges = _log_bins(min_pos, max_g, nbins_y)

    Z, N = _binned_stat_2d(ratings, games, sigma, x_edges, y_edges, agg="median")

    # Allow singletons; only mask completely empty bins
    # If grid is still too sparse, perform a gentle neighbor fill
    fill_ratio = np.isfinite(Z).mean() if Z.size else 0.0
    if fill_ratio < 0.50:
        Z = _iterative_nan_fill(Z, max_passes=6)

    # Smooth a bit (optional; if SciPy exists)
    Zs = _gaussian_smooth_nan(Z, sigma_xy=(1.0, 1.0), passes=1)

    # Build grids
    Xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    Yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(Xc, Yc)

    # Plot
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Zs, cmap="viridis", edgecolor="none", alpha=0.9, antialiased=True)

    # Scatter overlay (downsample to keep it responsive)
    idx = np.arange(ratings.size)
    if idx.size > 4000:
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, size=4000, replace=False)
    ax.scatter(ratings[idx], games[idx], sigma[idx], s=6, c="k", alpha=0.18)

    # Labels
    ax.set_xlabel("Rating")
    ax.set_ylabel("Games (log scale)")
    ax.set_zlabel("σ (uncertainty)")

    # Y tick labels in "game counts" style (mapped to bin centers)
    yticks = []
    ylabels = []
    for v in [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]:
        if v >= y_edges[0] and v <= y_edges[-1]:
            yticks.append(v); ylabels.append(str(int(v)))
    def _nearest(val, arr): arr = np.asarray(arr); return arr[np.argmin(np.abs(arr - val))]
    ax.set_yticks([_nearest(v, Yc) for v in yticks])
    ax.set_yticklabels(ylabels, rotation=0)

    cb = fig.colorbar(surf, shrink=0.6, aspect=18, pad=0.08)
    cb.set_label("σ (uncertainty) — lower is better")
    ax.set_title("Certainty vs Games vs Rating — median σ per bin\n(adaptive bins, neighbor fill, gentle smoothing)")
    ax.view_init(elev=28, azim=-45)
    plt.tight_layout()
    plt.show()
