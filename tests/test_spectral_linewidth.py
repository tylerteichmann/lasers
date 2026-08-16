import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from lasers import spectral_linewidth


def test_known_value() -> None:
    assert spectral_linewidth(168e-12, 777.783e-9) == pytest.approx(83.25553032941084e9)


def test_integer_input_returns_float_scalar() -> None:
    assert isinstance(spectral_linewidth(1, 1), np.float64)


@pytest.mark.parametrize(
    ("linewidth", "center_wavelength", "expected"),
    [(0.0, 632.8e-9, 0.0), (1e-12, 0.0, np.inf), (0.0, 0.0, np.nan)],
)
def test_zero_behavior(
    linewidth: float, center_wavelength: float, expected: float
) -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        assert_allclose(spectral_linewidth(linewidth, center_wavelength), expected)


@pytest.mark.parametrize(
    "wavelengths",
    [[500e-9, 632.8e-9, 1000e-9], np.array([500e-9, 632.8e-9, 1000e-9])],
)
def test_array_like_input(wavelengths: list[float] | NDArray[np.float64]) -> None:
    result = spectral_linewidth(1e-12, wavelengths)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert_allclose(result, [spectral_linewidth(1e-12, w) for w in wavelengths])
