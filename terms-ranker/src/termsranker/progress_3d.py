import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _idw_grid(x, y, z, grid_res=45, power=2.0, eps=1e-6):
    """
    Simple Inverse Distance Weighting interpolation onto a regular grid.
    x, y, z are 1D arrays. Returns Xg, Yg, Zg.
    """
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    # Small padding so surface doesn't touch frame
    xr = (x_max - x_min) * 0.05 or 1.0
    yr = (y_max - y_min) * 0.05 or 1.0
    xi = np.linspace(x_min - xr, x_max + xr, grid_res)
    yi = np.linspace(y_min - yr, y_max + yr, grid_res)
    Xg, Yg = np.meshgrid(xi, yi)

    # Compute IDW per grid point (vectorized)
    Zg = np.empty_like(Xg, dtype=float)
    for i in range(grid_res):
        # distances to all points for a row of grid Yg[i]
        dx = Xg[i, :, None] - x[None, :]
        dy = Yg[i, :, None] - y[None, :]
        dist2 = dx*dx + dy*dy
        weights = 1.0 / (np.power(dist2 + eps, power / 2.0))
        Zg[i, :] = (weights @ z) / np.sum(weights, axis=1)
    return Xg, Yg, Zg

def show_progress_3d(terms):
    if not terms:
        print("No terms loaded.")
        return

    ratings = np.array([t.rating for t in terms], dtype=float)
    games = np.array([max(1, t.games) for t in terms], dtype=float)
    sigma = np.array([t.sigma for t in terms], dtype=float)

    # X = rating, Y = log1p(games), Z = sigma
    X = ratings
    Y = np.log1p(games)
    Z = sigma

    Xg, Yg, Zg = _idw_grid(X, Y, Z, grid_res=50, power=1.8)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(Xg, Yg, Zg, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.scatter(X, Y, Z, s=10, c="k", alpha=0.3)  # original points (context)

    ax.set_title("Certainty vs. Activity vs. Rating", pad=16)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Games (log scale)")
    ax.set_zlabel("Sigma (uncertainty)")
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12)
    cbar.set_label("Sigma")

    ax.view_init(elev=25, azim=-135)
    plt.tight_layout()
    plt.show()
