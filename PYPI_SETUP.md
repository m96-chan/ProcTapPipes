# PyPI Publishing Setup Guide

Quick guide to set up automated publishing to PyPI and TestPyPI.

## 1. Create Accounts

### TestPyPI (for testing)
1. Go to https://test.pypi.org/account/register/
2. Create an account
3. Verify your email

### PyPI (production)
1. Go to https://pypi.org/account/register/
2. Create an account
3. Verify your email
4. Enable 2FA (required for publishing)

## 2. Configure Trusted Publishing

This is the **recommended** method - no API tokens needed!

### TestPyPI

1. Go to https://test.pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   ```
   PyPI Project Name: proctap-pipes
   Owner: YOUR_GITHUB_USERNAME (or organization)
   Repository name: ProcTapPipes
   Workflow name: publish-to-pypi.yml
   Environment name: testpypi
   ```
4. Click "Add"

### PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   ```
   PyPI Project Name: proctap-pipes
   Owner: YOUR_GITHUB_USERNAME (or organization)
   Repository name: ProcTapPipes
   Workflow name: publish-to-pypi.yml
   Environment name: pypi
   ```
4. Click "Add"

## 3. Configure GitHub Environments

1. Go to your GitHub repository
2. Settings → Environments
3. Click "New environment"
4. Create `testpypi` environment
5. Create `pypi` environment

### Optional: Add Protection for PyPI

For the `pypi` environment:
1. Click on the environment name
2. Add protection rules:
   - ✅ Required reviewers: Add yourself or team members
   - ✅ Deployment branches: Select "Selected branches" → Add `main`

This ensures production releases require approval.

## 4. Enable Workflow Permissions

1. Repository Settings → Actions → General
2. Scroll to "Workflow permissions"
3. Select "Read and write permissions"
4. Check "Allow GitHub Actions to create and approve pull requests"
5. Click "Save"

## 5. Test the Setup

### Method 1: Manual Test (TestPyPI only)

1. Go to Actions tab
2. Click "Publish to PyPI and TestPyPI"
3. Click "Run workflow"
4. Select:
   - Branch: `main`
   - Target: `testpypi`
5. Click "Run workflow"
6. Watch the workflow run
7. Check https://test.pypi.org/project/proctap-pipes/

### Method 2: Tag-based Release

```bash
# Create a test tag
git tag v0.1.0-test
git push origin v0.1.0-test

# Watch GitHub Actions
# Check TestPyPI and PyPI
```

## 6. Install and Test

### From TestPyPI

```bash
# Create a clean virtual environment
python3 -m venv test-env
source test-env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            proctap-pipes

# Test it
python -c "from proctap_pipes import WhisperPipe; print('Success!')"
proctap-whisper --help
```

### From PyPI

```bash
# Install from PyPI
pip install proctap-pipes

# Test it
python -c "from proctap_pipes import SlackWebhookPipe; print('Success!')"
```

## 7. First Real Release

Once everything is tested:

```bash
# Option A: Use the release script
./scripts/release.sh patch  # or minor, or major

# Option B: Manual
git tag v0.1.0
git push origin main
git push origin v0.1.0
```

## Troubleshooting

### "Invalid or non-existent authentication"

**Cause**: Trusted publishing not configured correctly

**Solution**:
1. Verify project name matches exactly: `proctap-pipes`
2. Check owner name matches your GitHub username/org
3. Ensure environment names match: `testpypi` and `pypi`
4. Wait a few minutes and retry

### "Version already exists on PyPI"

**Cause**: You're trying to upload the same version twice

**Solution**:
1. Increment version in `pyproject.toml`
2. Update `__version__` in `src/proctap_pipes/__init__.py`
3. Commit and create new tag

### "Permission denied"

**Cause**: Workflow doesn't have write permissions

**Solution**:
1. Settings → Actions → General
2. Enable "Read and write permissions"

### Workflow doesn't run

**Cause**: Tag format doesn't match

**Solution**:
- Tags must be in format: `v1.2.3`
- Must start with lowercase `v`
- Must be semantic version: `MAJOR.MINOR.PATCH`

## Alternative: API Token Method

If trusted publishing doesn't work, use API tokens:

### Create Tokens

1. **TestPyPI**:
   - Go to https://test.pypi.org/manage/account/token/
   - Click "Add API token"
   - Token name: "GitHub Actions"
   - Scope: "Entire account" or specific project
   - Copy the token (starts with `pypi-`)

2. **PyPI**:
   - Go to https://pypi.org/manage/account/token/
   - Same process as TestPyPI

### Add to GitHub Secrets

1. Repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add:
   - Name: `TEST_PYPI_API_TOKEN`
   - Value: `pypi-...` (token from TestPyPI)
4. Add:
   - Name: `PYPI_API_TOKEN`
   - Value: `pypi-...` (token from PyPI)

### Update Workflow

Edit `.github/workflows/publish-to-pypi.yml`:

Replace the publish steps with:

```yaml
- name: Publish to TestPyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    repository-url: https://test.pypi.org/legacy/
    password: ${{ secrets.TEST_PYPI_API_TOKEN }}

- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN }}
```

## Resources

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)

## Quick Reference

```bash
# Install development tools
pip install bump2version build twine

# Bump version locally
bump2version patch  # 0.1.0 → 0.1.1
bump2version minor  # 0.1.1 → 0.2.0
bump2version major  # 0.2.0 → 1.0.0

# Build package
python -m build

# Check package
twine check dist/*

# Test upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```
