import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from lasers import peak_fluence


def test_known_value() -> None:
    assert peak_fluence(100e-3, 1e-2) == pytest.approx(636.6197723675814)


def test_integer_input_returns_float_scalar() -> None:
    assert isinstance(peak_fluence(1, 1), np.float64)


@pytest.mark.parametrize(
    ("pulse_energy", "beam_radius", "expected"),
    [(1e-3, 0.0, np.inf), (0.0, 0.0, np.nan)],
)
def test_zero_behavior(
    pulse_energy: float, beam_radius: float, expected: float
) -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        assert_allclose(peak_fluence(pulse_energy, beam_radius), expected)


@pytest.mark.parametrize("radii", [[1e-2, 2e-2], np.array([1e-2, 2e-2])])
def test_array_like_input(radii: list[float] | NDArray[np.float64]) -> None:
    result = peak_fluence(1e-3, radii)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert_allclose(result, [peak_fluence(1e-3, r) for r in radii])
