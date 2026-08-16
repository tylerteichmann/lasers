"""Tests for spectral_linewidth module."""

import numpy as np
import pytest

from lasers import spectral_linewidth


def test_docstring_example():
    """Test the example from the docstring."""
    result = spectral_linewidth(168e-12, 777.783e-9)
    assert abs(result - 83.25553032941084e9) < 1e5


def test_proportional_to_linewidth():
    """Test that output scales linearly with linewidth."""
    result1 = spectral_linewidth(1e-12, 632.8e-9)
    result2 = spectral_linewidth(2e-12, 632.8e-9)
    assert abs(result2 - 2 * result1) < 1e5


def test_inverse_square_wavelength():
    """Test that output scales inversely with wavelength squared."""
    result1 = spectral_linewidth(1e-12, 500e-9)
    result2 = spectral_linewidth(1e-12, 1000e-9)
    # For double the wavelength, result should be 1/4
    assert abs(result1 - 4 * result2) < 1e5


def test_zero_wavelength_returns_inf():
    """Test that zero wavelength returns infinity."""
    result = spectral_linewidth(1e-12, 0)
    assert np.isinf(result)


def test_zero_linewidth():
    """Test that zero linewidth returns zero."""
    assert spectral_linewidth(0, 632.8e-9) == 0


def test_very_small_linewidth():
    """Test numerical stability with sub-picometer linewidth."""
    result = spectral_linewidth(1e-15, 632.8e-9)
    assert result > 0


def test_very_large_linewidth():
    """Test numerical stability with nanometer-scale linewidth."""
    result = spectral_linewidth(1e-9, 632.8e-9)
    assert result > 0


def test_array_input():
    """Test with numpy array wavelengths."""
    wavelengths = np.array([500e-9, 632.8e-9, 1000e-9])
    result = spectral_linewidth(1e-12, wavelengths)
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    assert np.all(result > 0)


def test_list_input():
    """Test with list input."""
    wavelengths = [500e-9, 632.8e-9, 1000e-9]
    result = spectral_linewidth(1e-12, wavelengths)
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
