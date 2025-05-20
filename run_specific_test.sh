#!/bin/bash
# Script to run a specific test with verbose output

# Check if a test path was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <test_path>"
    echo "Examples:"
    echo "  $0 tests/test_github_client.py"
    echo "  $0 tests/test_github_client.py::TestGitHubClientInit"
    echo "  $0 tests/test_github_client.py::TestGitHubClientInit::test_init_with_config"
    exit 1
fi

# Change to the project directory
cd "$(dirname "$0")"

# Check if the user wants coverage report
if [[ "$*" == *"--cov"* ]]; then
    COVERAGE_REQUESTED=true
else
    COVERAGE_REQUESTED=false
fi

# Check if pytest-cov is installed
if python -c "import pytest_cov" 2>/dev/null; then
    COV_INSTALLED=true
else
    COV_INSTALLED=false
    if [ "$COVERAGE_REQUESTED" = true ]; then
        echo "Warning: pytest-cov is not installed. Running tests without coverage."
        echo "To install: pip install pytest-cov"
    fi
fi

# Run the specified test with verbose output
if [ "$COVERAGE_REQUESTED" = true ] && [ "$COV_INSTALLED" = true ]; then
    python -m pytest "$1" -v --cov=src
else
    python -m pytest "$1" -v
fi

echo ""
echo "To run with coverage, add --cov to the command:"
echo "$0 $1 --cov"
