import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from lasers import m_squared


def test_exported_from_package() -> None:
    import lasers

    assert callable(lasers.m_squared)


def test_known_value() -> None:
    assert m_squared(7.5e-7, 1064e-9) == pytest.approx(2.2144685058198728)


def test_diffraction_limited_beam_is_unity() -> None:
    wavelength = 1064e-9
    assert m_squared(wavelength / np.pi, wavelength) == pytest.approx(1.0)


@pytest.mark.parametrize("factor", [2.0, 10.0, 0.5])
def test_proportional_to_bpp(factor: float) -> None:
    base = m_squared(7.5e-7, 1064e-9)
    assert m_squared(factor * 7.5e-7, 1064e-9) == pytest.approx(factor * base)


@pytest.mark.parametrize("factor", [2.0, 10.0, 0.5])
def test_inversely_proportional_to_wavelength(factor: float) -> None:
    base = m_squared(7.5e-7, 1064e-9)
    assert m_squared(7.5e-7, factor * 1064e-9) == pytest.approx(base / factor)


def test_zero_wavelength_returns_inf() -> None:
    with np.errstate(divide="ignore"):
        assert np.isinf(m_squared(7.5e-7, 0.0))


@pytest.mark.parametrize("bpps", [[1e-6, 2e-6], np.array([1e-6, 2e-6])])
def test_array_like_input(bpps: list[float] | NDArray[np.float64]) -> None:
    result = m_squared(bpps, 1064e-9)
    assert isinstance(result, np.ndarray)
    assert_allclose(result, [m_squared(b, 1064e-9) for b in bpps])
