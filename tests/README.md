# GitHub Stars Search Tests

This directory contains tests for the GitHub Stars Search project.

## Installation

To run the tests, you need to install the testing dependencies:

```bash
# Install all dependencies including testing
pip install -r requirements.txt
```

Or install just the testing dependencies:

```bash
# Using the provided script
./install_test_deps.sh

# Or manually
pip install pytest pytest-cov pytest-mock
```

## Test Structure

The tests are organized by module:

- `test_github_client.py`: Tests for the GitHub API client
- `test_content_processor.py`: Tests for the content processor
- `test_storage_manager.py`: Tests for the storage manager
- `test_embedding_manager.py`: Tests for the embedding manager
- `test_search_engine.py`: Tests for the search engine
- `test_cli_commands.py`: Tests for the CLI commands
- `test_cli_utils.py`: Tests for the CLI utilities
- `test_main.py`: Tests for the main script

## Running Tests

### Using the Test Scripts

The project includes scripts to simplify running tests:

```bash
# Run all tests with coverage report
./run_tests.sh

# Run all tests with HTML coverage report
./run_tests.sh --html

# Run a specific test with verbose output
./run_specific_test.sh tests/test_github_client.py
./run_specific_test.sh tests/test_github_client.py::TestGitHubClientInit
./run_specific_test.sh tests/test_github_client.py::TestGitHubClientInit::test_init_with_config

# Run tests with a specific marker
./run_marked_tests.sh unit
./run_marked_tests.sh integration
./run_marked_tests.sh "not api"

# Run tests with a specific marker and generate HTML coverage report
./run_marked_tests.sh unit --html
```

### Using pytest Directly

You can also run tests directly with pytest:

```bash
cd github_stars
pytest
```

To run a specific test file:

```bash
pytest tests/test_github_client.py
```

To run a specific test class:

```bash
pytest tests/test_github_client.py::TestGitHubClientInit
```

To run a specific test method:

```bash
pytest tests/test_github_client.py::TestGitHubClientInit::test_init_with_config
```

## Test Categories

Tests are categorized using pytest markers:

- `unit`: Unit tests that test individual components in isolation
- `integration`: Integration tests that test the interaction between components
- `api`: Tests that interact with the GitHub API (requires a valid API key)
- `slow`: Tests that are slow to run

To run tests with a specific marker:

```bash
pytest -m unit
```

To run tests excluding a specific marker:

```bash
pytest -m "not api"
```

## Test Coverage

To run tests with coverage:

```bash
pytest --cov=src
```

To generate a coverage report:

```bash
pytest --cov=src --cov-report=html
```

This will generate a coverage report in the `htmlcov` directory.

## Mocking

Most tests use mocking to avoid dependencies on external services. The `unittest.mock` module is used for this purpose.

For example, to mock the GitHub API client:

```python
@patch("src.api.github_client.Github")
def test_get_starred_repositories(self, mock_github):
    # Set up the mock
    mock_user = MagicMock()
    mock_repo = MagicMock()
    # ...
    
    # Call the method
    repos = github_client.get_starred_repositories()
    
    # Assertions
    assert len(repos) == 1
    # ...
```

## Test Data

Test data is defined at the top of each test file. For example:

```python
# Sample test data
SAMPLE_REPO = {
    "id": 12345,
    "name": "test-repo",
    "full_name": "test-user/test-repo",
    "description": "A test repository"
}
```

## Fixtures

Pytest fixtures are used to set up test dependencies. For example:

```python
@pytest.fixture
def mock_storage_manager():
    """Create a mock storage manager."""
    manager = MagicMock()
    
    # Mock the has_repository method
    manager.has_repository.return_value = False
    
    # ...
    
    return manager
```

## Adding New Tests

When adding new tests:

1. Create a new test file if testing a new module
2. Add test classes for each component or functionality
3. Add test methods for each behavior or scenario
4. Use appropriate markers to categorize tests
5. Use mocking to avoid dependencies on external services
6. Use fixtures to set up test dependencies
7. Use assertions to verify expected behavior
