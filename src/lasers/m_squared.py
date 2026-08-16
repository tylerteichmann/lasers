import numpy as np
from numpy.typing import ArrayLike, NDArray


def m_squared(
    beam_parameter_product: ArrayLike, wavelength: ArrayLike
) -> np.float64 | NDArray[np.float64]:
    b = np.asarray(beam_parameter_product, dtype=np.float64)
    lam = np.asarray(wavelength, dtype=np.float64)
    return (b * np.pi) / lam
