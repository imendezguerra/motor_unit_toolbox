# Motor Unit Toolbox

[![PyPI](https://img.shields.io/pypi/v/motor-unit-toolbox)](https://pypi.org/project/motor-unit-toolbox/)
[![Python versions](https://img.shields.io/pypi/pyversions/motor-unit-toolbox)](https://pypi.org/project/motor-unit-toolbox/)
[![CI](https://github.com/imendezguerra/motor_unit_toolbox/actions/workflows/ci.yml/badge.svg)](https://github.com/imendezguerra/motor_unit_toolbox/actions/workflows/ci.yml)
[![Docs](https://github.com/imendezguerra/motor_unit_toolbox/actions/workflows/docs.yml/badge.svg)](https://imendezguerra.github.io/motor_unit_toolbox/)

## Overview
<!-- --8<-- [start:overview] -->
Motor Unit Toolbox is a Python package to analyse motor unit (MU) behaviour, from computing basic firing and motor unit action potential (MUAP) properties, to comparing sets of spike trains and tracking MUAPs.

The package is composed of the following modules:

- `props`: MU properties such as discharge rate, pulse to noise ratio, silhouette measure, and coefficient of variation of the interspike intervals, as well as MUAP features.
- `spike_comp`: compare spike trains between paired or unpaired sets, as well as within sets. Main metrics are rate of agreement, precision, sensitivity, F1 score, true positives, false positives, and false negatives.
- `muap_comp`: compare, cluster and track MUAPs within or across recordings.
- `plots`: plot spike trains, MUAPs, and grouped MUAPs.
- `utils`: convert between lists of firing times and binary spike train matrices.
<!-- --8<-- [end:overview] -->

## Table of Contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Installation
<!-- --8<-- [start:install] -->
Install the latest release from PyPI (Python 3.10 or newer):

```sh
pip install motor-unit-toolbox
```

The package is imported as `motor_unit_toolbox`.
<!-- --8<-- [end:install] -->

To work on the code itself, see the development setup in [CONTRIBUTING.md](https://github.com/imendezguerra/motor_unit_toolbox/blob/main/CONTRIBUTING.md#development-setup).

## Quick start
<!-- --8<-- [start:quickstart] -->
Spike trains are binary matrices of shape `(samples, motor units)`. The example below builds two synthetic motor units, computes their firing properties and compares them with a second (shifted) decomposition:

```python
import numpy as np

from motor_unit_toolbox import props, spike_comp, utils

fs = 2048                                  # sampling frequency (Hz)
n_samples = 10 * fs                        # 10 s recording
timestamps = np.arange(n_samples) / fs

# Spike times (in samples) of two motor units firing at ~10 Hz and ~15 Hz
rng = np.random.default_rng(0)
firings = [
    np.cumsum(rng.normal(fs / 10, 10, size=95)).astype(int),
    np.cumsum(rng.normal(fs / 15, 10, size=140)).astype(int),
]
spike_trains = utils.firings_to_binary(firings, n_samples)  # (samples, units)

# Firing properties per motor unit
props.get_discharge_rate(spike_trains, timestamps)            # array([10.05, 15.18]) Hz
props.get_coefficient_of_variation(spike_trains, timestamps)  # array([0.047, 0.074])

# Agreement with a second decomposition of the same units (here: shifted by 2 samples)
roa, pairs, lags = spike_comp.rate_of_agreement_paired(
    spike_trains, np.roll(spike_trains, 2, axis=0), fs=fs
)
roa                                                           # array([1., 1.])
```
<!-- --8<-- [end:quickstart] -->

## Documentation
The full API reference, with every function and its arguments, is at
[imendezguerra.github.io/motor_unit_toolbox](https://imendezguerra.github.io/motor_unit_toolbox/).

## Contributing
Contributions are welcome! [CONTRIBUTING.md](https://github.com/imendezguerra/motor_unit_toolbox/blob/main/CONTRIBUTING.md) explains how to set up a development environment, run the tests, preview the docs, and what the automated checks (pre-commit and CI) do.

## License
This project is licensed under the [MIT License](https://github.com/imendezguerra/motor_unit_toolbox/blob/main/LICENSE).

## Citation

If you use this code in your research, please cite it. On GitHub, the **Cite this repository** button (from [`CITATION.cff`](https://github.com/imendezguerra/motor_unit_toolbox/blob/main/CITATION.cff)) gives APA and BibTeX formats, or use:

```bibtex
@software{Mendez_Guerra_Motor_Unit_Toolbox,
  author = {Mendez Guerra, Irene},
  title = {{Motor Unit Toolbox}},
  url = {https://github.com/imendezguerra/motor_unit_toolbox}
}
```

## Contact

For any questions or inquiries, please contact:
```
Irene Mendez Guerra
irene.mendez17@imperial.ac.uk
```
