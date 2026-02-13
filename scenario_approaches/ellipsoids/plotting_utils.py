import numpy as np
import matplotlib.pyplot as plt

def plot_ellipse(Q, c, r, ax, alpha=0.75, color='green', **kwargs):
    """
    Plot ellipse defined by (x-c)^T Q (x-c) <= r in 2D
    """
    eigvals, eigvecs = np.linalg.eigh(Q)
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    # Scale the ellipse by sqrt(r)
    ellipse = eigvecs @ np.diag(np.sqrt(r) / np.sqrt(eigvals)) @ circle
    ellipse = ellipse.T + c
    ax.plot(ellipse[:, 0], ellipse[:, 1], **kwargs, color=color, alpha=alpha)

    return ax
