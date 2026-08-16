import numpy as np
from numpy.typing import ArrayLike


def bpp(beam_waist: ArrayLike, divergence: ArrayLike) -> float | np.ndarray:
    beam_waist = np.asarray(beam_waist)
    divergence = np.asarray(divergence)
    return beam_waist * divergence
