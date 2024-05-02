# ANNUBeS

|     Badges     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| :------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  **fairness**  | [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8861/badge)](https://www.bestpractices.dev/projects/8861) [![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|  **package**   | [![PyPI version](https://badge.fury.io/py/annubes.svg)](https://badge.fury.io/py/annubes)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|    **docs**    | [![Documentation](https://img.shields.io/badge/docs-mkdocs-259482)](https://annubs.github.io/annubes/latest/) [![RSD](https://img.shields.io/badge/rsd-annubes-00a3e3.svg)](https://research-software-directory.org/projects/annubes) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11098460.svg)](https://doi.org/10.5281/zenodo.11098460)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|   **tests**    | [![build](https://github.com/ANNUBS/annubes/actions/workflows/build.yml/badge.svg)](https://github.com/ANNUBS/annubes/actions/workflows/build.yml) [![sonarcloud](https://github.com/ANNUBS/annubes/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/ANNUBS/annubes/actions/workflows/sonarcloud.yml) [![markdown-links](https://github.com/ANNUBS/annubes/actions/workflows/markdown-links.yml/badge.svg)](https://github.com/ANNUBS/annubes/actions/workflows/markdown-links.yml) [![cffconvert](https://github.com/ANNUBS/annubes/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/ANNUBS/annubes/actions/workflows/cffconvert.yml) [![linting](https://github.com/ANNUBS/annubes/actions/workflows/linting.yml/badge.svg)](https://github.com/ANNUBS/annubes/actions/workflows/linting.yml) [![static-typing](https://github.com/ANNUBS/annubes/actions/workflows/static-typing.yml/badge.svg)](https://github.com/ANNUBS/annubes/actions/workflows/static-typing.yml) [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=ANNUBS_annubes&metric=alert_status)](https://sonarcloud.io/dashboard?id=ANNUBS_annubes) [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=ANNUBS_annubes&metric=coverage)](https://sonarcloud.io/dashboard?id=ANNUBS_annubes) |
| **running on** | [![ubuntu](https://img.shields.io/badge/ubuntu-latest-8A2BE2?style=plastic)](https://github.com/actions/runner-images?tab=readme-ov-file#available-images) [![mac](https://img.shields.io/badge/macos-latest-8A2BE2?style=plastic)](https://github.com/actions/runner-images?tab=readme-ov-file#available-images) [![win](https://img.shields.io/badge/windows-latest-8A2BE2?style=plastic)](https://github.com/actions/runner-images?tab=readme-ov-file#available-images) [![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|  **license**   | [![github license badge](https://img.shields.io/github/license/ANNUBS/annubes)](https://github.com/ANNUBS/annubes?tab=Apache-2.0-1-ov-file)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

## Overview

The use of animals in neuroscience research is a fundamental tool to understand the inner workings of the brain during perception and cognition in health and disease. Neuroscientists train animals, often rodents, in behavioral tasks over several months, however training protocols are sometimes not well defined and this leads to delays in research, additional costs, or the need of more animals. Finding strategies to optimize animal training in safe and ethical ways is therefore of crucial importance in neuroscience.

ANNUBeS, which stays for _Artificial Neural Networks to Uncover Behavioral Strategies_, is a deep learning framework meant to generate synthetic data and train on them neural networks aimed at developing and evaluating animals' training protocols in neuroscience. The package gives the users the possibility to generate behavioral data in a very flexible way, that can be used to train neural networks in the same way that animals are trained, and study whether the developed models can predict the behavior of the animals. The ultimate goal of the framework is to lead researchers to more efficient training protocols, thus improving neuroscience practices.

📚 [Documentation](https://annubs.github.io/annubes/latest/)

🐛 Bugs reports and ⭐ features requests [here](https://github.com/ANNUBS/annubes/issues)

🔧 [Pull Requests](https://github.com/ANNUBS/annubes/pulls)

For more details about how to contribute, see the [contribution guidelines](CONTRIBUTING.md).

❗❗ DISCLAIMER ❗❗

Please note that this software is currently in its early stages of development. As such, some features may not work exactly as intended or envisioned yet. We appreciate your patience and understanding. If you encounter any issues or have suggestions for improvement, we encourage you to open an issue on our repository. Thank you for your support!

## Table of contents

- [ANNUBeS](#annubes)
  - [Overview](#overview)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
    - [Repository installation](#repository-installation)
    - [Pip installation](#pip-installation)
  - [Get started](#get-started)
    - [Generate synthetic data](#generate-synthetic-data)
    - [Train neural networks](#train-neural-networks)
  - [Contributing](#contributing)
  - [Credits](#credits)
  - [Package development](#package-development)

## Installation

### Repository installation

We advise to install the package inside a virtual environment (using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [venv](https://docs.python.org/3/library/venv.html)).

After having created and activated your environment, to install ANNUBeS from GitHub repository, do:

```console
git clone git@github.com:ANNUBS/annubes.git
cd annubes
pip install .
```

### Pip installation

Under development.

## Get started

### Generate synthetic data

The `Task` data class can be used for defining a behavioral task, and many parameters can be set. The configuration of the trials that can appear during a session is given by a dictionary representing the ratio of the different trials within the task (`session`). Trials with a single modality (e.g., a visual trial) must be represented by single characters, while trials with multiple modalities (e.g., an audiovisual trial) are represented by the character combination of those trials. The probability of catch trials (denoted by X) in the session can be set using the `catch_prob` parameter.

```python
from annubes.task import Task

task = Task(name='example_task',
                session={"v":0.5, "a":0.5},
                stim_intensities=[0.7, 0.9],
                stim_time=2000,
                catch_prob=0.3)
```

For more details about the `Task` class parameters, see the [API Documentation](https://annubs.github.io/annubes/latest/api/task/#annubes.task.Task).

Then, trials can be generated:

```python

NTRIALS = 10
trials = task.generate_trials(NTRIALS)
```

And plotted:

```python
task.plot_trials(NTRIALS)
```

<p align="center">
  <img src="https://github.com/ANNUBS/annubes/blob/ead1437b7ee6ad6998ce2b3653fd0b3b3d875e25/docs/example_trials_plot.png?raw=true" width="700">
</p>

### Train neural networks

This functionality is still under development.

## Contributing

If you want to contribute to the development of annubes,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).

## Package development

If you're looking for developer documentation, go [here](https://github.com/ANNUBS/annubes/blob/main/README.dev.md).
