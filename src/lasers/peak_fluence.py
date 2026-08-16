import numpy as np
from numpy.typing import ArrayLike, NDArray


def peak_fluence(
    pulse_energy: ArrayLike, beam_radius: ArrayLike
) -> np.float64 | NDArray[np.float64]:
    """Calculate the peak fluence of a laser pulse.

    Peak fluence is the on-axis energy density of a pulse with a Gaussian
    transverse profile, which is twice the average fluence over the beam area.

    Parameters
    ----------
    pulse_energy : float or array-like
        Energy of the laser pulse in joules.
    beam_radius : float or array-like
        Beam radius at the 1/e^2 intensity point, in meters.

    Returns
    -------
    float or np.ndarray
        Peak fluence in joules per square meter. Returns a scalar for scalar
        inputs, np.ndarray for array-like inputs. A zero beam radius returns
        inf.

    Examples
    --------
    >>> peak_fluence(100e-3, 1e-2)
    np.float64(636.6197723675814)
    """
    energy = np.asarray(pulse_energy, dtype=np.float64)
    w = np.asarray(beam_radius, dtype=np.float64)
    return (2.0 * energy) / (np.pi * (w * w))
