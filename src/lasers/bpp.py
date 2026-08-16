import numpy as np
from numpy.typing import ArrayLike, NDArray


def bpp(
    beam_waist: ArrayLike, divergence: ArrayLike
) -> np.float64 | NDArray[np.float64]:
    w = np.asarray(beam_waist, dtype=np.float64)
    theta = np.asarray(divergence, dtype=np.float64)
    return w * theta
