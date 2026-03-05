from math import pi


def peak_fluence(pulse_energy: float, beam_radius: float):
    """Calculate the peak fluence of a laser.

    Parameters
    ----------
    pulse_energy : float
        Energy of the laser pulse in joules.
    beam_radius : float
        Radius of the laser beam in centimeters.

    Returns
    -------
    float
        Peak fluence in joules per centimeter squared.

    Examples
    --------
    >>> peak_fluence(100e-3, 1.0)
    31830.98861837907
    """
    return (2.0 * pulse_energy) / (pi * (beam_radius * beam_radius))