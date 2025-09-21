from typing import Optional, Sequence, Tuple, Union
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

ArrayLike = Union[np.ndarray, Sequence[float]]

def _ensure_outdir(outdir: Optional[str]) -> None:
    if outdir:
        os.makedirs(outdir, exist_ok=True)

def _maybe_save(fig: Figure, outdir: Optional[str], fname: Optional[str]) -> None:
    if outdir and fname:
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")

def plot_timeseries(
    t: ArrayLike,
    y: ArrayLike,
    *,
    label: Optional[str] = None,
    title: str = "Time series",
    outdir: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False,
):
    """
    Plot y(t) for ODEs or a single probe of a PDE.

    Parameters
    ----------
    t : (N,) array
    y : (N,) array
    """
    _ensure_outdir(outdir)
    t = np.asarray(t)
    y = np.asarray(y)

    fig, ax = plt.subplots()
    ax.plot(t, y, lw=1.5, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(title)
    if label:
        ax.legend(frameon=False)
    ax.grid(True, ls=":", alpha=0.5)

    _maybe_save(fig, outdir, filename or "timeseries.png")
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_field_1d(
    x: ArrayLike,
    u: ArrayLike,
    *,
    title: str = "Field (1D)",
    xlabel: str = "x",
    ylabel: str = "u(x)",
    outdir: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False,
):
    """
    Plot a 1D spatial field u(x).
    """
    _ensure_outdir(outdir)
    x = np.asarray(x)
    u = np.asarray(u)

    fig, ax = plt.subplots()
    ax.plot(x, u, lw=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)

    _maybe_save(fig, outdir, filename or "field_1d.png")
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_field_2d(
    u: ArrayLike,
    *,
    x_extent: Optional[Tuple[float, float]] = None,
    y_extent: Optional[Tuple[float, float]] = None,
    title: str = "Field (2D)",
    cmap: Optional[str] = None,
    colorbar: bool = True,
    outdir: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False,
):
    """
    Plot a 2D field u(y, x) as an image.

    Parameters
    ----------
    u : (Ny, Nx) array
    x_extent : (xmin, xmax) to label axes (optional)
    y_extent : (ymin, ymax) to label axes (optional)
    """
    _ensure_outdir(outdir)
    u = np.asarray(u)
    extent = None
    if (x_extent is not None) and (y_extent is not None):
        extent = (x_extent[0], x_extent[1], y_extent[0], y_extent[1])

    fig, ax = plt.subplots()
    im = ax.imshow(
        u,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap or "viridis",
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)

    _maybe_save(fig, outdir, filename or "field_2d.png")
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_spacetime(
    x: ArrayLike,
    t: ArrayLike,
    U: ArrayLike,
    *,
    title: str = "Space–time (u(x,t))",
    cmap: Optional[str] = None,
    outdir: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False,
):
    """
    Space–time diagram for 1D PDE: U has shape (Nt, Nx) or (Nx, Nt).
    Detect orientation automatically and plot as image.
    """
    _ensure_outdir(outdir)
    x = np.asarray(x)
    t = np.asarray(t)
    U = np.asarray(U)

    # Try to get (Nt, Nx)
    if U.shape == (t.size, x.size):
        U_im = U
        extent = (x.min(), x.max(), t.min(), t.max())
        xlabel, ylabel = "x", "t"
    elif U.shape == (x.size, t.size):
        U_im = U.T
        extent = (x.min(), x.max(), t.min(), t.max())
        xlabel, ylabel = "x", "t"
    else:
        raise ValueError(f"U shape {U.shape} incompatible with x={x.size}, t={t.size}")

    fig, ax = plt.subplots()
    im = ax.imshow(
        U_im,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap or "viridis",
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)

    _maybe_save(fig, outdir, filename or "spacetime.png")
    if show:
        plt.show()
    else:
        plt.close(fig)