import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import numpy as np

def show_progress_3d(terms):
    """
    Open a matplotlib 3D surface showing how sigma (uncertainty)
    varies with rating and number of games.
    """
    if not terms:
        print("No terms loaded.")
        return

    ratings = np.array([t.rating for t in terms], dtype=float)
    games = np.array([max(1, t.games) for t in terms], dtype=float)
    sigma = np.array([t.sigma for t in terms], dtype=float)

    # Use log scale for games to spread low values
    x = ratings
    y = np.log1p(games)
    z = sigma

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(x, y, z, cmap="viridis", linewidth=0.2, antialiased=True)

    ax.set_title("Certainty vs. Activity vs. Rating", pad=16)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Games (log scale)")
    ax.set_zlabel("Sigma (uncertainty)")

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Sigma")

    ax.view_init(elev=25, azim=-135)
    plt.tight_layout()
    plt.show()
