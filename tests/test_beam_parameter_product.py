import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from lasers import beam_parameter_product


def test_exported_from_package() -> None:
    import lasers

    assert callable(lasers.beam_parameter_product)


def test_known_value() -> None:
    assert beam_parameter_product(1.5e-3, 0.5e-3) == pytest.approx(7.5e-7)


@pytest.mark.parametrize("factor", [2.0, 10.0, 0.5])
def test_proportional_to_beam_waist(factor: float) -> None:
    base = beam_parameter_product(1.5e-3, 0.5e-3)
    assert beam_parameter_product(factor * 1.5e-3, 0.5e-3) == pytest.approx(
        factor * base
    )


@pytest.mark.parametrize("factor", [2.0, 10.0, 0.5])
def test_proportional_to_divergence(factor: float) -> None:
    base = beam_parameter_product(1.5e-3, 0.5e-3)
    assert beam_parameter_product(1.5e-3, factor * 0.5e-3) == pytest.approx(
        factor * base
    )


def test_zero_waist_returns_zero() -> None:
    assert beam_parameter_product(0.0, 0.5e-3) == 0.0


@pytest.mark.parametrize("waists", [[1e-3, 2e-3, 3e-3], np.array([1e-3, 2e-3, 3e-3])])
def test_array_like_input(waists: list[float] | NDArray[np.float64]) -> None:
    result = beam_parameter_product(waists, 1e-3)
    assert isinstance(result, np.ndarray)
    assert_allclose(result, [beam_parameter_product(w, 1e-3) for w in waists])
