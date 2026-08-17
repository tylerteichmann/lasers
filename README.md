# lasers

A Python library of laser physics equations for researchers and engineers.

[![PyPI version](https://img.shields.io/pypi/v/lasers.svg)](https://pypi.org/project/lasers/)
[![Python](https://img.shields.io/pypi/pyversions/lasers.svg)](https://pypi.org/project/lasers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/tylerteichmann/lasers/actions/workflows/ci.yaml/badge.svg)](https://github.com/tylerteichmann/lasers/actions/workflows/ci.yaml)

## Installation

```bash
$ pip install lasers
```

## Usage

```python
import lasers as lz

dnu = lz.spectral_linewidth(linewidth=168e-12, center_wavelength=777.783e-9)
BPP = lz.beam_parameter_product(beam_waist=1.5e-3, divergence=0.5e-3)
m2 = lz.m_squared(beam_parameter_product=BPP, wavelength=1064e-9)
F = lz.peak_fluence(pulse_energy=100e-3, beam_radius=1e-2)
```

## Overview

`lasers` is a Python library for laser physicists and optical engineers, providing
well-documented implementations of equations and computational tools commonly
encountered in laser science and engineering.

The design philosophy prioritizes accessibility to scientists over software
convention. Function signatures and parameter names reflect the terminology
established in standard references — Siegman, Saleh & Teich, Boyd — so that
the interface reads naturally to anyone trained in the field. Inputs and outputs
use SI units throughout.

`lasers` is intended to serve both interactive and programmatic use cases.
Individual functions support rapid estimation and verification during
experimental work, while the broader API accommodates integration into parameter
studies, system-level models, and automated analysis pipelines.

The library is under active development. Its scope will continue to expand to
reflect the breadth of analytical methods used across the optics and photonics
community. Contributions, feature requests, and domain expertise are welcomed.

## Contributing

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/tylerteichmann/lasers/issues).

## Development

```bash
pip install -e ".[dev]"
```

## Citation

If you use this package in published work, please cite it:

```bibtex
@software{teichmann_lasers,
  author  = {Teichmann, Tyler},
  title   = {lasers: A Python library of laser physics equations},
  url     = {https://github.com/tylerteichmann/lasers},
  version = {0.0.2},
}
```

## Authors

[Tyler Teichmann](https://github.com/tylerteichmann)

## License

MIT — see [LICENSE](LICENSE).
