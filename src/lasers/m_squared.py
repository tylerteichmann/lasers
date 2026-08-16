from typing import Union

import numpy as np
from math import pi
from numpy.typing import ArrayLike


def m_squared(bpp: ArrayLike, wavelength: ArrayLike) -> Union[float, np.ndarray]:
    bpp = np.asarray(bpp)
    wavelength = np.asarray(wavelength)
    return (bpp * pi) / wavelength
