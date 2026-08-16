from math import pi

import numpy as np
from numpy.typing import ArrayLike


def m_squared(bpp: ArrayLike, wavelength: ArrayLike) -> float | np.ndarray:
    bpp = np.asarray(bpp)
    wavelength = np.asarray(wavelength)
    return (bpp * pi) / wavelength
