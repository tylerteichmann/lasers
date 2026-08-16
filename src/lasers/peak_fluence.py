import numpy as np
from numpy.typing import ArrayLike, NDArray


def peak_fluence(
    pulse_energy: ArrayLike, beam_radius: ArrayLike
) -> np.float64 | NDArray[np.float64]:
    energy = np.asarray(pulse_energy, dtype=np.float64)
    w = np.asarray(beam_radius, dtype=np.float64)
    return (2.0 * energy) / (np.pi * (w * w))
