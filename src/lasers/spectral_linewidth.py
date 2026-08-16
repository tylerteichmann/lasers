
import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import c


def spectral_linewidth(
    linewidth: ArrayLike, center_wavelength: ArrayLike
) -> float | np.ndarray:
    """Calculate the spectral linewidth of a laser.

    Converts wavelength domain linewidth to frequency domain using the
    relationship between frequency and wavelength.

    Parameters
    ----------
    linewidth : float or array-like
        Linewidth in meters (wavelength domain).
    center_wavelength : float or array-like
        Center wavelength in meters.

    Returns
    -------
    float or np.ndarray
        Spectral linewidth in Hz (frequency domain). Returns scalar for scalar
        inputs, np.ndarray for array-like inputs.

    Examples
    --------
    >>> spectral_linewidth(168e-12, 777.783e-9)
    83.25553032941084e9
    """
    linewidth = np.asarray(linewidth)
    center_wavelength = np.asarray(center_wavelength)
    return (c * linewidth) / (center_wavelength * center_wavelength)
