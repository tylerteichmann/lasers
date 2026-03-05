from scipy.constants import c


def spectral_linewidth(linewidth: float, center_wavelength: float) -> float:
    """Calculate the spectral linewidth of a laser.

    Converts wavelength domain linewidth to frequency domain using the
    relationship between frequency and wavelength.

    Parameters
    ----------
    linewidth : float
        Linewidth in meters (wavelength domain).
    center_wavelength : float
        Center wavelength in meters.

    Returns
    -------
    float
        Spectral linewidth in Hz (frequency domain).

    Examples
    --------
    >>> spectral_linewidth(168e-12, 777.783e-9)
    83.25553032941084e9
    """
    return (c * linewidth) / (center_wavelength * center_wavelength)