import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from lasers import spectral_linewidth


def test_known_value() -> None:
    assert spectral_linewidth(168e-12, 777.783e-9) == pytest.approx(83.25553032941084e9)


@pytest.mark.parametrize("factor", [2.0, 10.0, 0.5])
def test_proportional_to_linewidth(factor: float) -> None:
    base = spectral_linewidth(1e-12, 632.8e-9)
    assert spectral_linewidth(factor * 1e-12, 632.8e-9) == pytest.approx(factor * base)


def test_inverse_square_wavelength() -> None:
    result1 = spectral_linewidth(1e-12, 500e-9)
    result2 = spectral_linewidth(1e-12, 1000e-9)
    assert result1 == pytest.approx(4.0 * result2)


def test_zero_wavelength_returns_inf() -> None:
    with np.errstate(divide="ignore"):
        assert np.isinf(spectral_linewidth(1e-12, 0.0))


def test_zero_linewidth_returns_zero() -> None:
    assert spectral_linewidth(0.0, 632.8e-9) == 0.0


@pytest.mark.parametrize("linewidth", [1e-15, 1e-12, 1e-9])
def test_positive_across_linewidth_range(linewidth: float) -> None:
    assert spectral_linewidth(linewidth, 632.8e-9) > 0


@pytest.mark.parametrize(
    "wavelengths",
    [[500e-9, 632.8e-9, 1000e-9], np.array([500e-9, 632.8e-9, 1000e-9])],
)
def test_array_like_input(wavelengths: list[float] | NDArray[np.float64]) -> None:
    result = spectral_linewidth(1e-12, wavelengths)
    assert isinstance(result, np.ndarray)
    assert_allclose(result, [spectral_linewidth(1e-12, w) for w in wavelengths])
