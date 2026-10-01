# Motor Unit Toolbox

[![CI](https://github.com/imendezguerra/motor_unit_toolbox/actions/workflows/ci.yml/badge.svg)](https://github.com/imendezguerra/motor_unit_toolbox/actions/workflows/ci.yml)

## Overview
This repository contains functions to analyse motor unit (MU) behaviour, from computing basic firing and motor unit action potential (MUAP) properties, to comparing sets of spike trains and tracking MUAPs.

## Table of Contents
- [Installation](#installation)
- [Quick start](#quickstart)
- [Running tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation
To set up the project locally do the following:

1. Clone the repository:
    ```sh
    git clone https://github.com/imendezguerra/motor_unit_toolbox.git
    ```
2. Navigate to the project directory:
    ```sh
    cd motor_unit_toolbox
    ```
3. Create the conda environment from the `environment.yml` file:
    ```sh
    conda env create -f environment.yml
    ```
4. Activate the environment:
    ```sh
    conda activate motor_unit_toolbox
    ```
5. Install toolbox:
    ```
    pip install -e .
    ```

## Quick start
The package is composed of the following modules:
- `muap_comp.py`: Functions to compare and track MUAPs.
- `spike_comp.py`: Functions to compare spike trains between paired or unpaired sets, as well as within sets. Main metrics are rate of agreement, precision, sensitivity, F1 score, true positives, false positives, and false negatives.
- `props.py`: Functions to extract MU properties such as discharge rate, pulse to noise ratio, silhouette measure, and coefficient of variation of the interspike intervals, as well as MUAP features.
- `plots.py`: Functions to plot the spike trains, MUAPs, and grouped MUAPs.


## Running tests
Install the package with the development extras and run the test suite:

```sh
pip install -e ".[dev]"
pytest                      # full suite
pytest -m "not slow"        # skip the slower clustering/tracking tests
pytest --cov                # with a coverage report
```

Tests marked `xfail` document known bugs. When one is fixed, the test starts to pass and pytest reports it as a failure (`xfail_strict`) so the marker can be removed.

## Contributing
We welcome contributions! Here’s how you can contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/newfeature`).
3. Install the development tools and git hooks:
    ```sh
    pip install -e ".[dev]"
    pre-commit install
    ```
4. Add tests for your change under `tests/` and make sure `pytest` passes.
5. Commit your changes (`git commit -m 'Add some newfeature'`). The pre-commit hooks run ruff and basic file checks.
6. Push to the branch (`git push origin feature/newfeature`).
7. Open a pull request. CI runs the linters, the test suite on Python 3.9–3.13 (Linux, plus macOS and Windows), a minimum-dependency check, and a packaging check.

## License
This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite this repository:

```sh
@software{Mendez_Guerra_Motor_Unit_Toolbox,
author = {Mendez Guerra, Irene},
title = {{Motor Unit Toolbox}},
url = {https://github.com/imendezguerra/motor_unit_toolbox},
version = {1.0}
}
```
## Contact

For any questions or inquiries, please contact us at:
```sh
Irene Mendez Guerra
irene.mendez17@imperial.ac.uk
```
