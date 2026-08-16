import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from lasers import peak_fluence


def test_known_value() -> None:
    assert peak_fluence(100e-3, 1e-2) == pytest.approx(636.6197723675814)


@pytest.mark.parametrize("factor", [2.0, 10.0, 0.5])
def test_proportional_to_pulse_energy(factor: float) -> None:
    base = peak_fluence(100e-3, 1e-2)
    assert peak_fluence(factor * 100e-3, 1e-2) == pytest.approx(factor * base)


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_inverse_square_radius(factor: float) -> None:
    base = peak_fluence(100e-3, 1e-2)
    assert peak_fluence(100e-3, factor * 1e-2) == pytest.approx(base / factor**2)


def test_zero_radius_returns_inf() -> None:
    with np.errstate(divide="ignore"):
        assert np.isinf(peak_fluence(1e-3, 0.0))


@pytest.mark.parametrize("radii", [[1e-2, 2e-2], np.array([1e-2, 2e-2])])
def test_array_like_input(radii: list[float] | NDArray[np.float64]) -> None:
    result = peak_fluence(1e-3, radii)
    assert isinstance(result, np.ndarray)
    assert_allclose(result, [peak_fluence(1e-3, r) for r in radii])
