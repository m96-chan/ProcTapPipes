# Release Guide for ProcTapPipes

This document describes how to release new versions of ProcTapPipes to PyPI and TestPyPI.

## Prerequisites

1. **GitHub Repository Setup**
   - Repository must be on GitHub
   - GitHub Actions must be enabled

2. **PyPI Trusted Publishing Setup**
   
   For **TestPyPI**:
   - Go to https://test.pypi.org/manage/account/publishing/
   - Add a new publisher:
     - PyPI Project Name: `proctap-pipes`
     - Owner: `your-github-username` or `organization-name`
     - Repository name: `ProcTapPipes`
     - Workflow name: `publish-to-pypi.yml`
     - Environment name: `testpypi`
   
   For **PyPI**:
   - Go to https://pypi.org/manage/account/publishing/
   - Add a new publisher with the same settings but:
     - Environment name: `pypi`

3. **GitHub Environment Setup**
   - Go to your repository Settings → Environments
   - Create two environments:
     - `testpypi`
     - `pypi`
   - (Optional) Add protection rules for the `pypi` environment

## Release Process

### Method 1: Automated Release via Git Tag (Recommended)

This method automatically publishes to both TestPyPI and PyPI when you push a version tag.

```bash
# 1. Update version in pyproject.toml
vim pyproject.toml  # Update version = "0.2.0"

# 2. Update CHANGELOG (if you have one)
vim CHANGELOG.md

# 3. Commit changes
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"

# 4. Create and push tag
git tag v0.2.0
git push origin main
git push origin v0.2.0

# GitHub Actions will automatically:
# - Build the package
# - Publish to TestPyPI
# - Publish to PyPI
# - Create a GitHub Release
```

### Method 2: Manual Workflow Dispatch

Use this for testing or manual releases.

1. Go to your repository on GitHub
2. Click "Actions" tab
3. Select "Publish to PyPI and TestPyPI" workflow
4. Click "Run workflow"
5. Select target:
   - `testpypi` - Publish only to TestPyPI
   - `pypi` - Publish only to PyPI
   - `both` - Publish to both

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.1.0): New functionality, backwards compatible
- **PATCH** version (0.0.1): Bug fixes, backwards compatible

Examples:
- `v0.1.0` - Initial release
- `v0.1.1` - Bug fix
- `v0.2.0` - New feature (WhisperPipe enhancement)
- `v1.0.0` - Stable API, production ready

## Pre-release Versions

For alpha, beta, or release candidates:

```bash
# Alpha release
git tag v0.2.0a1
git push origin v0.2.0a1

# Beta release
git tag v0.2.0b1
git push origin v0.2.0b1

# Release candidate
git tag v0.2.0rc1
git push origin v0.2.0rc1
```

## Testing a Release

### Test on TestPyPI

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            proctap-pipes

# Test the installation
python -c "from proctap_pipes import WhisperPipe; print('Success!')"

# Test CLI tools
proctap-whisper --help
proctap-llm --help
proctap-webhook --help
```

### Test on PyPI

```bash
# Install from PyPI
pip install proctap-pipes

# Run tests
python -c "from proctap_pipes import SlackWebhookPipe; print('Success!')"
```

## Rollback

If you need to remove a bad release:

1. **PyPI**: You cannot delete releases, but you can yank them:
   ```bash
   # Using twine (requires API token)
   pip install twine
   twine yank proctap-pipes 0.2.0
   ```

2. **GitHub**: Delete the release and tag:
   ```bash
   # Delete GitHub release via web UI
   # Then delete the tag
   git tag -d v0.2.0
   git push origin :refs/tags/v0.2.0
   ```

## Troubleshooting

### Build Fails

Check the build output:
1. Go to Actions → Failed workflow
2. Expand "Build package" step
3. Fix errors in `pyproject.toml` or source files

### Publish Fails

Common issues:

1. **Version already exists**
   - Increment version number
   - TestPyPI: Workflow will skip existing versions

2. **Trusted Publishing not configured**
   - Follow "PyPI Trusted Publishing Setup" above
   - Ensure environment names match exactly

3. **Permission denied**
   - Check repository → Settings → Actions → General
   - Ensure "Read and write permissions" is enabled

### GitHub Release Fails

- Ensure `contents: write` permission is set
- Check if tag already has a release
- Verify `gh` CLI commands in the workflow

## Manual Release (Fallback)

If GitHub Actions is unavailable:

```bash
# 1. Install build tools
pip install build twine

# 2. Build package
python -m build

# 3. Check package
twine check dist/*

# 4. Upload to TestPyPI
twine upload --repository testpypi dist/*

# 5. Upload to PyPI
twine upload dist/*
```

## Post-Release Checklist

After a successful release:

- [ ] Verify package on PyPI: https://pypi.org/project/proctap-pipes/
- [ ] Test installation: `pip install proctap-pipes`
- [ ] Update documentation if needed
- [ ] Announce release (Twitter, Discord, etc.)
- [ ] Close milestone on GitHub (if using milestones)
- [ ] Create new milestone for next version

## Resources

- [PyPI Publishing Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [GitHub Actions for PyPI](https://github.com/marketplace/actions/pypi-publish)
- [Semantic Versioning](https://semver.org/)
- [Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
