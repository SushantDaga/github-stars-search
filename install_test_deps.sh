#!/bin/bash
# Script to install testing dependencies

echo "Installing testing dependencies..."
pip install pytest pytest-cov pytest-mock

echo ""
echo "Testing dependencies installed. You can now run the tests:"
echo "./run_tests.sh"
