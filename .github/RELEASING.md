# Releasing

Maintainer notes for publishing a new version to [PyPI](https://pypi.org/project/motor-unit-toolbox/)
and keeping the CI/CD setup up to date. Day-to-day development checks (pre-commit, `ci.yml`,
the docs build) are described in [`CONTRIBUTING.md`](../CONTRIBUTING.md#automated-checks).

- [Overview](#overview)
- [Publishing: `publish.yml`](#publishing-publishyml)
- [Docs deploy: `docs.yml`](#docs-deploy-docsyml)
- [One-off setup](#one-off-setup)
- [Release checklist](#release-checklist)
- [Maintenance](#maintenance)
- [When a release fails](#when-a-release-fails)

## Overview

Releases are driven by GitHub. Publishing a GitHub release, or clicking *Run workflow*, starts:

| Event | `ci.yml` | `docs.yml` | `publish.yml` |
|---|---|---|---|
| Publish a GitHub release | ✅ (called by publish) | build + **deploy** | upload to **PyPI** |
| *Run workflow* button | ✅ (called by publish) | build + **deploy** | upload to **TestPyPI** |

> **Versions are permanent.** PyPI never accepts the same version twice, even after deleting it.
> A broken release can only be *yanked* and followed by a new version. Rehearse on TestPyPI first.

## Publishing: `publish.yml`

**Purpose:** turn a GitHub release into a PyPI release, with no manual uploads and no stored
passwords.

**Triggers:**
- **Publish a GitHub release:** uploads to PyPI.
- ***Run workflow***: uploads to TestPyPI, for rehearsal.

The job chain is below. Each job only starts if the previous one succeeded:

```
test ──► build ──┬──► publish-pypi (release only, waits for approval) ──► attach-to-release
                 └──► publish-testpypi (manual run only)
```

| Job | What it does |
|---|---|
| **test** | Runs all of `ci.yml`. If anything fails, nothing is uploaded |
| **build** | Builds the wheel and sdist once and runs `twine check --strict`. On releases, it also checks that the tag (`v1.2.0`) matches `__version__` and the version in `CITATION.cff`. Stores `dist/` as an artifact so the later jobs upload exactly these files |
| **publish-testpypi** | Uploads to test.pypi.org through the `testpypi` environment |
| **publish-pypi** | Pauses until a maintainer **approves** the `pypi` environment, then uploads to pypi.org |
| **attach-to-release** | Adds the wheel and sdist to the GitHub release page |

**How uploading works without a password (Trusted Publishing):**
1. The upload job asks GitHub for a short-lived signed token, which is why it needs
   `id-token: write`.
2. The token states "repo imendezguerra/motor_unit_toolbox, workflow publish.yml, environment pypi".
3. PyPI checks that statement against the trusted publisher registered in the project
   settings, and accepts the upload.
4. The publish action also signs the files (attestations), so users can check where they came from.

## Docs deploy: `docs.yml`

On a published release, or *Run workflow*, the docs job builds the site with
`mkdocs build --strict`. It then publishes it to https://imendezguerra.github.io/motor_unit_toolbox/
through the `github-pages` environment.

Deploying on releases rather than on every push to `main` keeps the site describing the
version that is on PyPI. A manual run from a feature branch is blocked by the `github-pages`
environment rules (only `main` and `v*` tags are allowed).

## One-off setup

1. **Accounts:** create accounts with two-factor authentication on
   [pypi.org](https://pypi.org) and [test.pypi.org](https://test.pypi.org). They are separate sites.
2. **Trusted publishers:** on each site, go to *Your projects → Publishing → Add a new pending
   publisher* (GitHub tab):

    | Field | Value |
    |---|---|
    | PyPI project name | `motor-unit-toolbox` |
    | Owner | `imendezguerra` |
    | Repository name | `motor_unit_toolbox` |
    | Workflow name | `publish.yml` |
    | Environment name | `pypi` (on PyPI) / `testpypi` (on TestPyPI) |

    A pending publisher does not reserve the name. The project is created by the first upload.

3. **GitHub environments:** in *Settings → Environments*, create `pypi` and `testpypi`.
   Add yourself as a **required reviewer** on `pypi`, so every upload waits for your approval.
   Leave *Prevent self-review* off, or you cannot approve your own release.
   Under *Deployment branches and tags* on `pypi`, choose *Selected branches and tags* and add
   the **tag** rule `v*`. Releases run on the tag, not on a branch, so a branch-only rule blocks
   the upload. `testpypi` can stay unrestricted, so you can rehearse from any branch.
4. **GitHub Pages:** in *Settings → Pages*, set *Source* to **GitHub Actions**.
   This creates a `github-pages` environment that only allows `main`. In *Settings → Environments
   → github-pages*, add the **tag** rule `v*` next to `main`, or the docs deploy on release fails.
5. **Zenodo (optional):** sign in to [Zenodo](https://zenodo.org) with GitHub and switch on
   this repository. Every GitHub release is then archived with a citable DOI, using the
   metadata in `CITATION.cff`. Add the DOI to `CITATION.cff` and the README afterwards.

## Release checklist

1. **Choose the version** using [semantic versioning](https://semver.org):
    - `MAJOR` for breaking changes, such as removing or renaming a function.
    - `MINOR` for new features.
    - `PATCH` for bug fixes.
2. **Bump the version** in two places, in a PR to `main`:
    - `motor_unit_toolbox/__init__.py` (`__version__`), which `pyproject.toml` reads.
    - `CITATION.cff` (`version:`), and set `date-released:` to the release date.
3. **Merge** once CI and the docs build are green.
4. **Rehearse on TestPyPI:** *Actions → publish → Run workflow* on `main`. Then, in a fresh
   environment:

    ```sh
    python -m venv /tmp/mut-test && source /tmp/mut-test/bin/activate
    pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ motor-unit-toolbox
    python -c "import motor_unit_toolbox; print(motor_unit_toolbox.__version__)"
    ```

    TestPyPI also refuses a version it has already seen. To rehearse twice, use a pre-release
    version such as `1.3.0rc1`.

5. **Release:** *Releases → Draft a new release*:
    - Create the tag `vX.Y.Z` on `main`.
    - Write the release notes, flagging any deprecations.
    - Click **Publish release**.
6. **Approve** the `pypi` deployment when the workflow pauses at *Upload to PyPI*.
7. **Verify** the release:
    - `pip install motor-unit-toolbox` installs the new version.
    - The [PyPI page](https://pypi.org/project/motor-unit-toolbox/) and the docs show it.

## Maintenance

Dependabot is off, so these updates are manual:

- **Action versions:** in all three workflows (`actions/checkout@v7` and so on), check the actions'
  GitHub release pages a few times a year. Keep `upload-artifact` and `download-artifact` on
  majors that were released together. Today that is v7 and v8.
- **Hook versions:** run `pre-commit autoupdate`, then `pre-commit run --all-files`.
- **Python versions (each October, when a new Python is released and an old one reaches end of
  life):**
    - Update the test matrix in `ci.yml`.
    - Update the classifiers and `requires-python` in `pyproject.toml`.
    - Update the ruff `target-version` and the `minimum-deps` Python version.
- **Minimum dependencies:** when you raise a lower bound in `pyproject.toml`, raise the matching
  pin in `ci/constraints-min.txt`.
- **MkDocs:** capped at `<2` because MkDocs 2.0 drops plugins (mkdocstrings).
  [Zensical](https://zensical.org) reads `mkdocs.yml` and is the likely migration path.
- **Deprecations:** `get_alignmnent` (misspelled alias of `get_alignment`) is due for removal in 2.0.

## When a release fails

- **Tests fail in the `test` job:** nothing was uploaded. Fix the problem on `main` and publish
  a new release. Delete the failed release and its tag first if you want to reuse the version
  number.
- **Version check fails:** the tag and the version in the code disagree, and nothing was
  uploaded. Delete the release and its tag, fix the version on `main`, and release again.
- **Upload fails with an OIDC / "invalid publisher" error:** the trusted publisher on PyPI
  doesn't match exactly. Check the workflow file name, the environment name and the
  repository owner/name.
- **The run waits forever at *Upload to PyPI*:** it is waiting for approval. Open the run and
  click *Review deployments*.
- **Upload blocked by environment rules:** the `pypi` environment is missing the `v*` tag rule
  (one-off setup, step 3).
- **Docs deploy fails on release:** the `github-pages` environment is missing the `v*` tag rule
  (one-off setup, step 4).
- **A bad version reached PyPI:** on PyPI, *Manage → Releases → Yank* it, then fix the problem
  and release a new patch version.
