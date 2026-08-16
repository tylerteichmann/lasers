import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from lasers import beam_parameter_product


def test_known_value() -> None:
    assert beam_parameter_product(1.5e-3, 0.5e-3) == pytest.approx(7.5e-7)


def test_integer_input_returns_float_scalar() -> None:
    assert isinstance(beam_parameter_product(2, 3), np.float64)


@pytest.mark.parametrize("waists", [[1e-3, 2e-3, 3e-3], np.array([1e-3, 2e-3, 3e-3])])
def test_array_like_input(waists: list[float] | NDArray[np.float64]) -> None:
    result = beam_parameter_product(waists, 1e-3)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert_allclose(result, [beam_parameter_product(w, 1e-3) for w in waists])
