# ANNUBeS

| fair-software.eu recommendations |                                                                                                                                                                                                                                 |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| (1/5) code repository            | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/ANNUBS/annubes)                                                                              |
| (2/5) license                    | [![github license badge](https://img.shields.io/github/license/ANNUBS/annubes)](https://github.com/ANNUBS/annubes)                                                                                                              |
| (3/5) community registry         | [![RSD](https://img.shields.io/badge/rsd-annubes-00a3e3.svg)](https://research-software-directory.org/projects/annubes)                                                                                                         |
| (4/5) citation                   | [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>)                                                                                                               |
| (5/5) checklist                  | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                        | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)                                                 |
| **Other best practices**         | &nbsp;                                                                                                                                                                                                                          |
| Static analysis                  | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=ANNUBS_annubes&metric=alert_status)](https://sonarcloud.io/dashboard?id=ANNUBS_annubes)                                                         |
| Coverage                         | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=ANNUBS_annubes&metric=coverage)](https://sonarcloud.io/dashboard?id=ANNUBS_annubes)                                                             |
| Documentation                    | [![Documentation Status](https://readthedocs.org/projects/annubes/badge/?version=latest)](https://annubes.readthedocs.io/en/latest/?badge=latest)                                                                               |
| **GitHub Actions**               | &nbsp;                                                                                                                                                                                                                          |
| Build                            | [![build](https://github.com/ANNUBS/annubes/actions/workflows/build.yml/badge.svg)](https://github.com/ANNUBS/annubes/actions/workflows/build.yml)                                                                              |
| Citation data consistency        | [![cffconvert](https://github.com/ANNUBS/annubes/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/ANNUBS/annubes/actions/workflows/cffconvert.yml)                                                               |
| SonarCloud                       | [![sonarcloud](https://github.com/ANNUBS/annubes/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/ANNUBS/annubes/actions/workflows/sonarcloud.yml)                                                               |
| MarkDown link checker            | [![markdown-link-check](https://github.com/ANNUBS/annubes/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/ANNUBS/annubes/actions/workflows/markdown-link-check.yml)                                    |

## Overview

The use of animals in neuroscience research is a fundamental tool to understand the inner workings of the brain during perception and cognition in health and disease. Neuroscientists train animals, often rodents, in behavioral tasks over several months, however training protocols are sometimes not well defined and this leads to delays in research, additional costs, or the need of more animals. Finding strategies to optimize animal training in safe and ethical ways is therefore of crucial importance in neuroscience.

ANNUBeS, which stays for _Artificial Neural Networks to Uncover Behavioral Strategies_, is a deep learning framework meant to generate synthetic data and train on them neural networks aimed at developing and evaluating animals' training protocols in neuroscience. The package gives the users the possibility to generate behavioral data in a very flexible way, that can be used to train neural networks in the same way that animals are trained, and study whether the developed models can predict the behavior of the animals. The ultimate goal of the framework is to lead researchers to more efficient training protocols, thus improving neuroscience practices.

📚 [Documentation](https://annubs.github.io/annubes/latest/)

🐛 Bugs reports and ⭐ features requests [here](https://github.com/ANNUBS/annubes/issues)

🔧 [Pull Requests](https://github.com/ANNUBS/annubes/pulls)

For more details about how to contribute, see the [contribution guidelines](CONTRIBUTING.md).

## Table of contents

- [ANNUBeS](#annubes)
  - [Overview](#overview)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Get started](#get-started)
    - [Generate synthetic data](#generate-synthetic-data)
    - [Train neural networks](#train-neural-networks)
  - [Contributing](#contributing)
  - [Credits](#credits)
  - [Package development](#package-development)

## Installation

We advise to install the package inside a virtual environment (using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [venv](https://docs.python.org/3/library/venv.html)).

After having created and activated your environment, to install ANNUBeS from GitHub repository, do:

```console
git clone git@github.com:ANNUBS/annubes.git
cd annubes
pip install .
```

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
  <img src="./docs/example_trials_plot.png" width="700">
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
