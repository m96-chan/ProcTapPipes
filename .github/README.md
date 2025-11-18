# GitHub Actions Workflows

This directory contains automated workflows for ProcTapPipes.

## Workflows

### 1. `publish-to-pypi.yml` - Package Publishing

**Triggers:**
- Automatically when a version tag is pushed (e.g., `v0.1.0`)
- Manually via workflow dispatch

**What it does:**
1. Builds the Python package
2. Publishes to TestPyPI
3. Publishes to PyPI (production)
4. Creates a GitHub Release with artifacts

**Manual Usage:**
```bash
# Via GitHub UI:
Actions → Publish to PyPI and TestPyPI → Run workflow → Select target

# Via git tag:
git tag v0.2.0
git push origin v0.2.0
```

### 2. `test.yml` - Continuous Testing

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

**What it does:**
1. Runs tests with pytest
2. Checks code formatting (black)
3. Lints code (ruff)
4. Type checks (mypy)
5. Uploads coverage reports

**Test Matrix:**
- Python versions: 3.10, 3.11, 3.12

### 3. `version-bump.yml` - Automated Version Bumping

**Triggers:**
- Manual workflow dispatch only

**What it does:**
1. Bumps version in `pyproject.toml` and `__init__.py`
2. Creates a git commit
3. Creates and pushes a git tag

**Usage:**
```bash
# Via GitHub UI:
Actions → Version Bump → Run workflow → Select bump type (patch/minor/major)
```

## Setup Instructions

### 1. PyPI Trusted Publishing

#### TestPyPI
1. Go to https://test.pypi.org/manage/account/publishing/
2. Add publisher:
   - Project: `proctap-pipes`
   - Owner: `your-github-username`
   - Repository: `ProcTapPipes`
   - Workflow: `publish-to-pypi.yml`
   - Environment: `testpypi`

#### PyPI (Production)
1. Go to https://pypi.org/manage/account/publishing/
2. Add publisher with same settings:
   - Environment: `pypi`

### 2. GitHub Environments

1. Go to Repository Settings → Environments
2. Create `testpypi` environment
3. Create `pypi` environment
4. (Optional) Add protection rules for `pypi`:
   - Required reviewers
   - Deployment branches: only `main`

### 3. Repository Settings

1. Settings → Actions → General
2. Workflow permissions:
   - ✅ Read and write permissions
   - ✅ Allow GitHub Actions to create and approve pull requests

### 4. Secrets (If not using Trusted Publishing)

If you prefer API tokens instead of trusted publishing:

1. Settings → Secrets and variables → Actions
2. Add secrets:
   - `PYPI_API_TOKEN` - PyPI token
   - `TEST_PYPI_API_TOKEN` - TestPyPI token

## Release Workflow

### Quick Release
```bash
# 1. Bump version using workflow
# Actions → Version Bump → Run → Select type

# 2. Tag is automatically pushed, triggering publish workflow

# 3. Monitor Actions tab for progress
```

### Manual Release
```bash
# 1. Update version
vim pyproject.toml  # Update version
vim src/proctap_pipes/__init__.py  # Update __version__

# 2. Commit and tag
git add .
git commit -m "chore: bump version to 0.2.0"
git tag v0.2.0
git push origin main
git push origin v0.2.0

# 3. Workflow runs automatically
```

## Troubleshooting

### Publish fails: "Version already exists"
- Increment version number
- For TestPyPI: workflow skips existing versions

### Publish fails: "Invalid or non-existent authentication"
- Check Trusted Publishing configuration
- Verify environment names match exactly
- Ensure repository settings allow trusted publishing

### Tests fail
- Check Python version compatibility
- Review failed test output in Actions tab
- Run tests locally: `pytest`

### Version bump fails
- Check if tag already exists
- Ensure write permissions are enabled
- Verify .bumpversion.cfg syntax

## Badge Integration

Add these badges to your README.md:

```markdown
[![PyPI version](https://badge.fury.io/py/proctap-pipes.svg)](https://badge.fury.io/py/proctap-pipes)
[![Tests](https://github.com/your-username/ProcTapPipes/actions/workflows/test.yml/badge.svg)](https://github.com/your-username/ProcTapPipes/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/your-username/ProcTapPipes/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/ProcTapPipes)
```

## Best Practices

1. **Testing before release**
   - Always publish to TestPyPI first
   - Test installation from TestPyPI
   - Only then push to PyPI

2. **Version management**
   - Follow semantic versioning
   - Use version bump workflow for consistency
   - Tag format: `v{major}.{minor}.{patch}`

3. **Security**
   - Use Trusted Publishing (no API tokens needed)
   - Protect `pypi` environment with required reviewers
   - Review changes before releasing

4. **Documentation**
   - Update CHANGELOG.md before release
   - Include release notes in GitHub Release
   - Document breaking changes clearly

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [Python Packaging Guide](https://packaging.python.org/)
