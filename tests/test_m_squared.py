import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from lasers import m_squared


def test_known_value() -> None:
    assert m_squared(7.5e-7, 1064e-9) == pytest.approx(2.2144685058198728)


def test_diffraction_limited_beam_is_unity() -> None:
    wavelength = 1064e-9
    assert m_squared(wavelength / np.pi, wavelength) == pytest.approx(1.0)


def test_integer_input_returns_float_scalar() -> None:
    assert isinstance(m_squared(1, 1), np.float64)


@pytest.mark.parametrize(
    ("beam_parameter_product", "wavelength", "expected"),
    [(7.5e-7, 0.0, np.inf), (0.0, 0.0, np.nan)],
)
def test_zero_behavior(
    beam_parameter_product: float, wavelength: float, expected: float
) -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        assert_allclose(m_squared(beam_parameter_product, wavelength), expected)


@pytest.mark.parametrize("bpps", [[1e-6, 2e-6], np.array([1e-6, 2e-6])])
def test_array_like_input(bpps: list[float] | NDArray[np.float64]) -> None:
    result = m_squared(bpps, 1064e-9)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert_allclose(result, [m_squared(b, 1064e-9) for b in bpps])
