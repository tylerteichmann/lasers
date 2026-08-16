import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import c


def spectral_linewidth(
    linewidth: ArrayLike, center_wavelength: ArrayLike
) -> np.float64 | NDArray[np.float64]:
    dlam = np.asarray(linewidth, dtype=np.float64)
    lam = np.asarray(center_wavelength, dtype=np.float64)
    return (c * dlam) / (lam * lam)
