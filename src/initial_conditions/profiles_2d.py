import numpy as np

def gaussian_bump_2d(X, Y, center=(0.0, 0.0), sigma=(0.2, 0.2), amp=1.0):
    """
    Isotropic or anisotropic 2D Gaussian bump.
    X, Y: meshgrid arrays.
    center: tuple (x_center, y_center)
    sigma: scalar or tuple (sigma_x, sigma_y)
    amp: amplitude
    """
    x0, y0 = center
    if np.isscalar(sigma):
        sx = sy = sigma
    else:
        sx, sy = sigma
    return amp * np.exp(-((np.square(X - x0) / (2 * np.square(sx))) + (np.square(Y - y0) / (2 * np.square(sy)))))

def square_pulse_2d(X, Y, center=(0.0, 0.0), width=(1.0, 1.0), amp=1.0):
    """
    2D square/rectangular pulse centered at center with given width (width_x, width_y).
    X, Y: meshgrid arrays.
    center: tuple (x_center, y_center)
    width: scalar or tuple (width_x, width_y)
    amp: amplitude
    """
    x0, y0 = center
    if np.isscalar(width):
        wx = wy = width
    else:
        wx, wy = width
    return amp * (
        ((X > x0 - np.divide(wx, 2.0)) & (X < x0 + np.divide(wx, 2.0))) &
        ((Y > y0 - np.divide(wy, 2.0)) & (Y < y0 + np.divide(wy, 2.0)))
    ).astype(float)

def triangle_wave_2d(X, Y, center=(0.0, 0.0), width=(1.0, 1.0), amp=1.0):
    """
    Separable 2D triangle wave: product of two 1D triangle waves in x and y.
    X, Y: meshgrid arrays.
    center: tuple (x_center, y_center)
    width: scalar or tuple (width_x, width_y)
    amp: amplitude
    """
    def triangle_1d(x, center, width, amp):
        slope = np.divide(2 * amp, width)
        return np.clip(amp - np.multiply(slope, np.abs(x - center)), 0, amp)
    x0, y0 = center
    if np.isscalar(width):
        wx = wy = width
    else:
        wx, wy = width
    return triangle_1d(X, x0, wx, amp) * triangle_1d(Y, y0, wy, amp)