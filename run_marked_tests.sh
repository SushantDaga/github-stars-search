#!/bin/bash
# Script to run tests with a specific marker

# Check if a marker was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <marker> [--html]"
    echo "Examples:"
    echo "  $0 unit"
    echo "  $0 integration"
    echo "  $0 \"not api\""
    echo "  $0 unit --html"
    exit 1
fi

# Check if HTML report is requested
if [ "$2" == "--html" ]; then
    HTML_REPORT=true
else
    HTML_REPORT=false
fi

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

# Run the tests with the specified marker
if [ "$HTML_REPORT" = true ] && [ "$COV_INSTALLED" = true ]; then
    python -m pytest -m "$1" --cov=src --cov-report=html
    
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
    python -m pytest -m "$1" --cov=src
else
    python -m pytest -m "$1"
fi
