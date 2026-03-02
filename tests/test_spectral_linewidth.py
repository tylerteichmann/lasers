"""Tests for spectral_linewidth module."""

import pytest
from lasers import spectral_linewidth


class TestSpectralLinewidth:
    """Tests for the spectral_linewidth function."""

def test_docstring_example():
    result = spectral_linewidth(168e-12, 777.783e-9)
    assert abs(result - 83.25553032941084e9) < 1e5

def test_proportional_to_linewidth():
    result1 = spectral_linewidth(1e-12, 632.8e-9)
    result2 = spectral_linewidth(2e-12, 632.8e-9)
    assert abs(result2 - 2 * result1) < 1e5

def test_inverse_square_wavelength():
    result1 = spectral_linewidth(1e-12, 500e-9)
    result2 = spectral_linewidth(1e-12, 1000e-9)
    assert abs(result1 - 4 * result2) < 1e5

def test_zero_wavelength_raises_error():
    with pytest.raises(ZeroDivisionError):
        spectral_linewidth(1e-12, 0)

def test_zero_linewidth():
    assert spectral_linewidth(0, 632.8e-9) == 0