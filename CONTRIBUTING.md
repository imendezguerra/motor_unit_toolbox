# Contributing

Thanks for helping improve Motor Unit Toolbox! This guide covers setting up a development
environment, the checks that run on your machine and on GitHub, and what to do when one fails.

- [Development setup](#development-setup)
- [Making a change](#making-a-change)
- [Running tests](#running-tests)
- [Documentation](#documentation)
- [Automated checks](#automated-checks)
  - [Local git hook: pre-commit](#local-git-hook-pre-commit)
  - [CI: `ci.yml`](#ci-ciyml)
  - [Docs build: `docs.yml`](#docs-build-docsyml)
- [When a check fails](#when-a-check-fails)

Maintainers: the release process is in [`.github/RELEASING.md`](.github/RELEASING.md).

## Development setup

1. Fork the repository on GitHub, then clone your fork and move into it:
    ```sh
    git clone https://github.com/<your-username>/motor_unit_toolbox.git
    cd motor_unit_toolbox
    ```
2. Create and activate the conda environment. It installs the toolbox in editable mode with the
   test, lint and docs tools listed in `pyproject.toml`:
    ```sh
    conda env create -f environment.yml
    conda activate motor_unit_toolbox
    ```
    Without conda, use a virtual environment and `pip install -e ".[dev,docs]"` instead.
3. Install the git hook (see [pre-commit](#local-git-hook-pre-commit)):
    ```sh
    pre-commit install
    ```

## Making a change

1. Create a feature branch: `git checkout -b feature/newfeature`.
2. Make your change and add tests for it under `tests/`.
3. Make sure `pytest` passes.
4. Commit: `git commit -m "Add some newfeature"`. The pre-commit hook checks the staged files first.
5. Push the branch (`git push origin feature/newfeature`) and open a pull request.
6. CI and the docs build run on the PR automatically. A PR is ready to merge when all checks are green.

If you change a function's arguments, update its docstring too. The API reference is generated
from the docstrings, and the docs build fails when they disagree with the signature.

## Running tests

```sh
pytest                      # full suite
pytest -m "not slow"        # skip the slower clustering/tracking tests
pytest --cov                # with a coverage report
```

Tests marked `xfail` document known bugs. When one is fixed, the test starts to pass and pytest
reports it as a failure (`xfail_strict`), so the marker can be removed.

Warnings are turned into errors (`filterwarnings = error` in `pyproject.toml`). Use
`pytest.warns(...)` in tests that expect a warning.

## Documentation

The site (https://imendezguerra.github.io/motor_unit_toolbox/) is built with MkDocs:

- `docs/`: the pages.
- `docs/api/`: the API reference, generated from the docstrings (Google style).
- `README.md`: the overview, install and quick-start sections are pulled in from here, between
  the `<!-- --8<-- [start:...] -->` markers.

To preview it locally, run `mkdocs serve` and open the address it prints
(http://127.0.0.1:8000/motor_unit_toolbox/). It reloads when you edit `docs/`, `README.md` or a
docstring.

## Automated checks

There are two kinds of automation, and they run in different places:

- **The git hook (pre-commit)** runs **on your machine** during `git commit`.
- **GitHub Actions workflows** (`.github/workflows/*.yml`) run **on GitHub's servers** when you
  push or open a PR. Results appear as checks on the PR and in the repo's **Actions** tab.

| Event | pre-commit (local) | `ci.yml` | `docs.yml` |
|---|---|---|---|
| `git commit` on your machine | ✅ on staged files | | |
| Push to a PR / open a PR | | ✅ | build |
| Push / merge to `main` | | ✅ | build |

Releases use two more steps, `publish.yml` and the docs deploy, described in
[`.github/RELEASING.md`](.github/RELEASING.md).

### Local git hook: pre-commit

Configured in `.pre-commit-config.yaml`. `pre-commit install` writes `.git/hooks/pre-commit`, so
the hooks run on every `git commit`, on the **staged** files only:

- **File hygiene:** strips trailing whitespace and makes files end with a newline. It also
  validates YAML and TOML syntax, blocks leftover merge-conflict markers and files over 1 MB,
  and blocks leftover `breakpoint()` / `pdb` calls.
- **`ruff check --fix`:** the linter, using the rules in `pyproject.toml` (`[tool.ruff]`).

If a hook fails or fixes something, **the commit is aborted**. Review the changes, `git add`
them, and commit again.

```sh
pre-commit run --all-files   # check the whole repo, not just staged files
git commit --no-verify       # skip the hooks once (CI will still run them)
```

### CI: `ci.yml`

**Purpose:** prove that every change keeps the package working, before it reaches `main`.

**Triggers:** every PR, every push to `main`, and manual runs. The release workflow also reuses it.

**Run control:** a newer push to the same branch cancels the run still in progress. The
workflow only has read access to the repository.

| Job | What it does | Catches |
|---|---|---|
| **Lint (pre-commit)** | Runs the same pre-commit hooks, on all files | Style/lint errors, commits made with `--no-verify` |
| **Test** (matrix) | `pip install -e ".[dev]"` then `pytest --cov`, on Linux for Python 3.10, 3.11, 3.12, 3.13 and 3.14, plus macOS and Windows on 3.12 | Bugs, Python-version and OS-specific breakage. Fails if coverage < 80%. The coverage table appears in the run summary |
| **Minimum deps** | Python 3.10 with the oldest supported versions pinned in `ci/constraints-min.txt` | Code that silently needs a newer numpy/scipy/etc. than `pyproject.toml` claims |
| **Build** | Builds the sdist and wheel, runs `twine check --strict`, checks that the sdist ships the test suite, then installs the wheel in a clean environment and imports every module | Packaging mistakes: missing files, broken metadata, or a README that PyPI cannot render |

Jobs run in parallel, and one failing Python version does not cancel the others.

### Docs build: `docs.yml`

On every PR and push to `main` it runs `mkdocs build --strict`. Strict mode fails on any warning,
such as a broken link, a missing README snippet, or a docstring whose arguments do not match the
function signature. The site is only *deployed* when a release is published.

## When a check fails

- **The commit is aborted by pre-commit:** the hook either fixed files itself (`git add` them and
  commit again) or printed the ruff errors to fix.
- **Lint is red in CI:** run `pre-commit run --all-files` locally and commit the result.
- **A test fails on one Python version only:** reproduce it in an environment with that version.
  With [uv](https://docs.astral.sh/uv/):
    ```sh
    uv venv -p 3.X .venv-3X && source .venv-3X/bin/activate
    uv pip install -e ".[dev]" && pytest
    ```
- **Minimum deps fails:** the change uses a feature newer than the lowest version allowed in
  `pyproject.toml`. Either avoid it, or raise the lower bound in `pyproject.toml` and the matching
  pin in `ci/constraints-min.txt`.
- **Coverage below 80%:** add tests for the new code paths. The CI run summary lists the
  uncovered lines.
- **Build fails:** run `python -m build` and `twine check --strict dist/*` locally (needs
  `pip install build twine`).
- **Docs build fails:** run `mkdocs build --strict` locally. The warning names the file and line,
  often a docstring argument that does not match the signature.
