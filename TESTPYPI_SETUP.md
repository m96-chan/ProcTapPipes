# TestPyPI Trusted Publishing Setup

## Problem
The workflow is failing to publish to TestPyPI with a "400 Bad Request" error. This is because **Trusted Publishing** needs to be configured on TestPyPI.

## Solution: Configure Trusted Publishing on TestPyPI

### Step 1: Create the package on TestPyPI (first time only)

Since the package doesn't exist on TestPyPI yet, you need to either:

**Option A: Manual first upload**
1. Build locally: `python -m build`
2. Upload to TestPyPI using a token:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```
   - You'll need a TestPyPI API token (create one at https://test.pypi.org/manage/account/token/)

**Option B: Skip TestPyPI and only use production PyPI**
- Remove or disable the TestPyPI job in the workflow
- Only publish to production PyPI

### Step 2: Configure Trusted Publishing on TestPyPI

1. Go to https://test.pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in the form:
   - **PyPI Project Name**: `proctap-pipes`
   - **Owner**: `m96-chan` (your GitHub username/org)
   - **Repository name**: `ProcTapPipes`
   - **Workflow name**: `publish-to-pypi.yml`
   - **Environment name**: `testpypi` (must match the `environment.name` in the workflow)
4. Click "Add"

### Step 3: Verify GitHub Environment

1. Go to your GitHub repo → Settings → Environments
2. Ensure an environment named `testpypi` exists
3. If not, the workflow will create it automatically on first run

### Step 4: Test the Publishing

After configuring trusted publishing:

```bash
# Create a new version tag
git tag v0.2.2
git push origin v0.2.2
```

This will trigger the workflow to publish to both TestPyPI and PyPI.

## Alternative: Disable TestPyPI

If you don't need TestPyPI for testing, you can disable it by modifying `.github/workflows/publish-to-pypi.yml`:

```yaml
publish-to-testpypi:
  name: Publish to TestPyPI
  if: false  # Disable this job
  # ... rest of the job
```

## Current Status

- ✅ PyPI: Working (version 0.2.0 published)
- ❌ TestPyPI: Not configured (package doesn't exist, trusted publishing not set up)
- ✅ Version: Synced to 0.2.2 across all files

## Next Steps

1. Choose Option A or B from Step 1 above
2. If choosing Option A, follow Steps 2-4
3. Create and push the v0.2.2 tag to trigger publication

## References

- [PyPI Trusted Publishing Documentation](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions OIDC](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
