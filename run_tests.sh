#!/bin/bash
# Script to run tests with coverage

# Change to the project directory
cd "$(dirname "$0")"

# Check if pytest-cov is installed
if python -c "import pytest_cov" 2>/dev/null; then
    COV_INSTALLED=true
else
    COV_INSTALLED=false
    echo "Warning: pytest-cov is not installed. Running tests without coverage."
    echo "To install: pip install pytest-cov"
fi

# Run tests
if [ "$1" == "--html" ] && [ "$COV_INSTALLED" = true ]; then
    # Generate HTML coverage report
    python -m pytest --cov=src --cov-report=html
    
    # Open the coverage report in the default browser
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        open htmlcov/index.html
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        # Linux
        if [ -x "$(command -v xdg-open)" ]; then
            xdg-open htmlcov/index.html
        else
            echo "Coverage report generated at htmlcov/index.html"
        fi
    elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ] || [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
        # Windows
        start htmlcov/index.html
    else
        echo "Coverage report generated at htmlcov/index.html"
    fi
elif [ "$COV_INSTALLED" = true ]; then
    # Run tests with terminal coverage report
    python -m pytest --cov=src
else
    # Run tests without coverage
    python -m pytest
fi

# Print help message
echo ""
echo "Run with --html to generate an HTML coverage report:"
echo "./run_tests.sh --html"
echo ""
echo "Run specific test categories with pytest markers:"
echo "pytest -m unit"
echo "pytest -m integration"
echo "pytest -m \"not api\""
