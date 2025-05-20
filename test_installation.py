#!/usr/bin/env python3
"""
Test script to verify the GitHub Stars Search installation.
"""

import os
import sys
from pathlib import Path

def check_module(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def main():
    """Run installation tests."""
    print("Testing GitHub Stars Search installation...")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # Check required modules
    # beautifulsoup4 is bs4 and pyyaml is yaml and scikit-learn is sklearn
    required_modules = [
        "requests", "github", "langdetect", "markdown", "bs4", 
        "html2text", "sentence_transformers", "txtai", "torch", "numpy", # sentence transformers have to be before txtai otherwise we get OMP error
        "sklearn", "click", "yaml", "tqdm", "colorama", "rich", "joblib"
    ]
    
    print("\nChecking required modules:")
    all_modules_found = True
    for module in required_modules:
        if check_module(module):
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module} (not found)")
            all_modules_found = False
    
    # Check project structure
    print("\nChecking project structure:")
    base_dir = Path(__file__).parent
    
    # Check directories
    directories = [
        "data", "data/repositories", "data/embeddings", "data/index",
        "logs", "src", "src/api", "src/cli", "src/embeddings",
        "src/processor", "src/search", "src/storage"
    ]
    
    all_dirs_found = True
    for directory in directories:
        dir_path = base_dir / directory
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ (not found)")
            all_dirs_found = False
    
    # Check key files
    key_files = [
        "github_stars_search.py", "config.yaml", "requirements.txt",
        "README.md", ".gitignore", "setup.py", "LICENSE"
    ]
    
    all_files_found = True
    print("\nChecking key files:")
    for file in key_files:
        file_path = base_dir / file
        if file_path.exists() and file_path.is_file():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not found)")
            all_files_found = False
    
    # Check environment
    print("\nChecking environment:")
    if os.environ.get("GITHUB_STARS_KEY"):
        print("  ✓ GITHUB_STARS_KEY environment variable")
    else:
        print("  ✗ GITHUB_STARS_KEY environment variable (not found)")
        print("    Make sure to set it in .env file")
    
    # Summary
    print("\nSummary:")
    if all_modules_found and all_dirs_found and all_files_found:
        print("✅ All checks passed! The installation looks good.")
    else:
        print("❌ Some checks failed. Please fix the issues above.")

if __name__ == "__main__":
    main()
