#!/bin/bash
# Release script for ProcTapPipes
# Usage: ./scripts/release.sh [patch|minor|major]

set -e

BUMP_TYPE="${1:-patch}"

if [[ ! "$BUMP_TYPE" =~ ^(patch|minor|major)$ ]]; then
    echo "Error: Invalid bump type. Use: patch, minor, or major"
    exit 1
fi

echo "Starting release process..."
echo ""

# Check if working directory is clean
if [[ -n $(git status -s) ]]; then
    echo "Error: Working directory is not clean"
    echo "Please commit or stash your changes first"
    git status -s
    exit 1
fi

# Pull latest changes
echo "Pulling latest changes..."
git pull origin main

# Run tests
echo "Running tests..."
if command -v pytest &> /dev/null; then
    pytest || {
        echo "Tests failed. Aborting release."
        exit 1
    }
else
    echo "pytest not found, skipping tests"
fi

# Get current version
CURRENT_VERSION=$(grep 'version = ' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "Current version: $CURRENT_VERSION"

# Bump version
if command -v bump2version &> /dev/null; then
    echo "Bumping $BUMP_TYPE version..."
    bump2version "$BUMP_TYPE" --verbose
    
    NEW_VERSION=$(grep 'version = ' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
    echo "New version: $NEW_VERSION"
else
    echo "bump2version not found. Install with: pip install bump2version"
    exit 1
fi

# Push
echo "Pushing to origin..."
git push origin main
git push origin "v$NEW_VERSION"

echo ""
echo "Release initiated!"
echo ""
echo "Next steps:"
echo "  1. Check GitHub Actions"
echo "  2. Monitor TestPyPI"
echo "  3. Check PyPI"
echo "  4. Verify GitHub Release"
echo ""
