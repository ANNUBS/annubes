# `annubes` developer documentation

If you're looking for user documentation, go [here](README.md).

## Code editor

We use [Visual Studio Code (VS Code)](https://code.visualstudio.com/) as code editor.
The VS Code settings for this project can be found in [.vscode](.vscode).
The settings will be automatically loaded and applied when you open the project with VS Code.
See [the guide](https://code.visualstudio.com/docs/getstarted/settings) for more info about workspace settings of VS Code.

## Project setup

The project has been originally setup using the [Netherlands eScience Center python template](https://github.com/NLeSC/python-template). Most of the choices or the current setup are explained in our [guide](https://guide.esciencecenter.nl). For a quick reference on software development, we refer to [the software guide checklist](https://guide.esciencecenter.nl/#/best_practices/checklist).

## Development install

```shell
# Create a virtual environment, e.g. with
python -m venv env

# activate virtual environment
source env/bin/activate

# make sure to have a recent version of pip and setuptools
python -m pip install --upgrade pip setuptools

# (from the project root directory)
# install annubes as an editable package
python -m pip install --no-cache-dir --editable .
# install development dependencies
python -m pip install --no-cache-dir --editable .[dev]
```

Afterwards check that the install directory is present in the `PATH` environment variable.

## Running the tests

There are two ways to run tests.

The first way requires an activated virtual environment with the development tools installed:

```shell
pytest -v
```

The second is to use `tox`, which can be installed separately (e.g. with `pip install tox`), i.e. not necessarily inside the virtual environment you use for installing `annubes`, but then builds the necessary virtual environments itself by simply running:

```shell
tox
```

Testing with `tox` allows for keeping the testing environment separate from your development environment.
The development environment will typically accumulate (old) packages during development that interfere with testing; this problem is avoided by testing with `tox`.

### Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine how much of the package's code is actually executed during tests.
In an activated virtual environment with the development tools installed, inside the package directory, run:

```shell
coverage run
```

This runs tests and stores the result in a `.coverage` file.
To see the results on the command line, run

```shell
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.

## Linting and formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting, sorting imports and formatting of python (notebook) files. The configurations of `ruff` are set in [ruff.toml](ruff.toml) file.

If you are using VS code, please install and activate the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) to automatically format and check linting.

Otherwise, please ensure check both linting (`ruff check .`) and formatting (`ruff format .`) before requesting a review. Running the linters and formatters requires an activated virtual environment with the development tools installed.

## Static typing

We use [Mypy](http://mypy-lang.org/) as static type checker:

```
# install mypy
pip install mypy

# run mypy
mypy path-to-source-code
```

Mypy configurations are set in [pyproject.toml](pyproject.toml) file.

For more info about static typing and mypy, see:

- [Static typing with Python](https://typing.readthedocs.io/en/latest/index.html#)
- [Mypy doc](https://mypy.readthedocs.io/en/stable/)

## Docs

We use [MkDocs](https://www.mkdocs.org/) and its theme [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
to generate documentations. The configurations of MkDocs are set in [mkdocs.yml](mkdocs.yml) file.

To watch the changes of current doc in real time, run:

```shell
mkdocs serve
# or to watch src and docs directories
mkdocs serve -w docs -w src
```

Then open your browser and go to `http://127.0.0.1:8000/`.

### Publishing the docs

The docs are published on GitHub Pages. We use [mike](https://github.com/jimporter/mike)
to deploy the docs to the `gh-pages` branch and to manage the versions of docs.

For example, to deploy the version 2.0 of the docs to the `gh-pages` branch and make it the latest
version, run:

```shell
mike deploy -p -u 2.0 latest
```

If you are not happy with the changes you can run `mike delete [version]`.
All these mike operations will be recorded as git commits of branch `gh-pages`.

`mike serve` is used to check all versions committed to branch `gh-pages`, which is for checking
the production website. If you have changes but not commit them yet, you should use `mkdocs serve`
instead of `mike serve` to check them.

## Versioning

Bumping the version across all files is done before creating a new package release, running `bump2version [part]` from command line after having installed [bump2version](https://pypi.org/project/bump2version/) on your local environment. Instead of `[part]`, type the part of the version to increase, e.g. minor. The settings in `.bumpversion.cfg` will take care of updating all the files containing version strings.

## Making a release

This section describes how to make a release in 3 parts:

1. preparation
2. making a release on GitHub
3. making a release on PyPI

### (1/3) Preparation

1. Verify that the information in `CITATION.cff` is correct, and that `.zenodo.json` contains equivalent data
2. [Bump the version](https://github.com/DeepRank/deeprank2/blob/dev/README.dev.md#versioning).
3. Run the unit tests with `pytest -v`

### (2/3) GitHub

Make a [release on GitHub](https://github.com/ANNUBS/annubes/releases/new).

### (3/3) PyPI

The GitHub release will trigger [a GitHub action](https://github.com/DeepRank/deeprank2/actions/workflows/release.yml) that will take care of publishing the package on PyPi.

## Development conventions

- Branching
  - When creating a new branch, please use the following convention: `<issue_number>_<description>_<author_name>`.
  - Always branch from `dev` branch, unless there is the need to fix an undesired status of `main`. See above for more details about the branching workflow adopted.
- Pull Requests
  - When creating a pull request, please use the following convention: `<type>: <description>`. Example _types_ are `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others based on the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines).
