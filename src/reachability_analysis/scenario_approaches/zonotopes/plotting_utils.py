import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.spatial import ConvexHull

def plot_zonotope_projection(center, G, ax=None, alpha=0.5, dim_x=0, dim_y=1, color="red"):
    """
    Plot the projection of a zonotope Z = { G @ xi | xi in [-1,1]^m }
    into 2D given by (dim_x, dim_y).

    Parameters
    ----------
    G : ndarray, shape (n, m)
        Generator matrix (n-dim zonotope with m generators).
    dim_x, dim_y : int
        Indices of the dimensions to project onto (0-based).
    ax : matplotlib Axes, optional
        If provided, draw into this axes.
    color : str
        Fill color of the polygon.
    """
    n, m = G.shape

    if center is None:
        center = np.zeros(n)

        # Project to 2D
    G2 = G[[dim_x, dim_y], :]  # shape (2, m)
    c2 = center[[dim_x, dim_y]]  # shape (2,)

    # Enumerate all 2^m vertices
    verts = []
    for signs in product([-1, 1], repeat=m):
        xi = np.array(signs)
        point = c2 + G2 @ xi
        verts.append(point)
    verts = np.array(verts)

    # Convex hull
    hull = ConvexHull(verts)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(verts[hull.vertices, 0], verts[hull.vertices, 1],
            color=color, alpha=alpha)
    v = hull.vertices
    ax.plot([verts[v[-1], 0], verts[v[0], 0]],
        [verts[v[-1], 1], verts[v[0], 1]],
        color=color, alpha=alpha)
    # ax.plot(verts[:, 0], verts[:, 1], "ro", markersize=2)
    # ax.plot(c2[0], c2[1], "r*", markersize=12, label="center")

    # ax.set_xlabel(f"dim {dim_x }")
    # ax.set_ylabel(f"dim {dim_y }")
    # ax.set_title(f"Zonotope projection to dims ({dim_x }, {dim_y })")
    # ax.set_aspect("equal")
    # ax.legend()
    return ax