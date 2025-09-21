from __future__ import annotations

import os
from typing import Iterable, Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def animate_1d(
    u_history: Sequence[np.ndarray] | np.ndarray,
    x: np.ndarray,
    *,
    times: Optional[Sequence[float]] = None,   # optional t_k per frame
    dt: Optional[float] = None,                # or a constant step if no times[]
    interval: int = 50,                        # ms between frames when previewing
    filename: Optional[str] = None,            # if set, write MP4 (or GIF)
    fps: int = 20,
    dpi: int = 120,
    stride: int = 1,                           # keep every stride-th frame
    title: Optional[str] = None,
    ylim: Optional[tuple[float, float]] = None,
    linewidth: float = 2.0,
    show: bool = True,                         # show in notebook
) -> animation.FuncAnimation:
    """
    Animate a 1D field u(x,t) from a time history.

    Parameters
    ----------
    u_history : list/array of shape (T, N) or list of (N,)
        Time snapshots of the solution. Can be list or np.ndarray.
    x : array of shape (N,)
        Spatial grid.
    times : optional list of shape (T,)
        Time stamps for each frame. If None, uses dt or frame index.
    dt : optional float
        Time step (used only if times is None).
    interval : int
        Delay between frames in milliseconds during interactive playback.
    filename : str or None
        If provided, saves an MP4 (if ffmpeg available) or a GIF fallback.
    fps : int
        Frames per second when saving.
    dpi : int
        Dots per inch for saved video.
    stride : int
        Subsample frames to speed up / keep output shorter.
    title : str or None
        Figure title.
    ylim : (ymin, ymax) or None
        Fixed y-limits. If None, computed from data with a small margin.
    linewidth : float
        Line width for the curve.
    show : bool
        Whether to display the animation inline (notebooks).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object (useful for notebooks).
    """
    # --- prepare frames ---
    U = np.asarray(u_history)
    if U.ndim == 1:
        U = U[None, :]  # single frame -> shape (1, N)
    if stride > 1:
        U = U[::stride]

    T_frames, N = U.shape
    if x.shape[0] != N:
        raise ValueError(f"x has length {x.shape[0]} but frames have N={N}")

    # time labels
    if times is not None:
        t_arr = np.asarray(times)
        if stride > 1:
            t_arr = t_arr[::stride]
        if t_arr.shape[0] != T_frames:
            raise ValueError("times length must match number of frames after stride.")
    else:
        if dt is not None:
            t_arr = np.arange(T_frames, dtype=float) * dt
        else:
            t_arr = np.arange(T_frames, dtype=float)

    # y-limits
    if ylim is None:
        ymin = float(np.nanmin(U))
        ymax = float(np.nanmax(U))
        if not np.isfinite([ymin, ymax]).all():
            ymin, ymax = -1.0, 1.0
        if np.isclose(ymin, ymax):
            pad = 1.0 if ymax == 0 else 0.05 * abs(ymax)
            ymin, ymax = ymin - pad, ymax + pad
        margin = 0.03 * (ymax - ymin)
        ylim = (ymin - margin, ymax + margin)

    # --- figure & artists ---
    fig, ax = plt.subplots()
    line, = ax.plot(x, U[0], lw=linewidth)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    if title:
        ax.set_title(title)
    time_txt = ax.text(
        0.02, 0.95,
        _format_time_label(t_arr[0]),
        transform=ax.transAxes,
        va="top", ha="left"
    )
    ax.grid(True, alpha=0.25)

    # --- update function ---
    def _update(i: int):
        line.set_ydata(U[i])
        time_txt.set_text(_format_time_label(t_arr[i]))
        return (line, time_txt)

    anim = animation.FuncAnimation(
        fig, _update, frames=T_frames, interval=interval, blit=True
    )

    # --- save if requested ---
    if filename:
        _save_animation(anim, filename, fps=fps, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return anim


def _format_time_label(t: float) -> str:
    # Pretty time label
    if abs(t) < 1e-9:
        return "t = 0"
    if abs(t) < 1e-3:
        return f"t = {t:.6f}"
    if abs(t) < 1.0:
        return f"t = {t:.4f}"
    return f"t = {t:.3f}"


def _save_animation(anim: animation.FuncAnimation, filename: str, *, fps: int, dpi: int) -> None:
    ext = os.path.splitext(filename)[1].lower()
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    # Try ffmpeg MP4 first
    try:
        if ext in (".mp4", ""):
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=fps, metadata={"artist": "PDE_solver"}, bitrate=-1)
            out = filename if ext == ".mp4" else f"{filename}.mp4"
            anim.save(out, writer=writer, dpi=dpi)
            return
    except Exception:
        pass

    # Fallback to GIF if Pillow is available
    try:
        from matplotlib.animation import PillowWriter
        out = filename if ext == ".gif" else f"{filename}.gif"
        anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
        return
    except Exception as e:
        raise RuntimeError(
            "Could not save animation. Install ffmpeg for MP4 or Pillow for GIF."
        ) from e