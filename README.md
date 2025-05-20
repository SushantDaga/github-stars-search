# GitHub Stars Search

Neural search for your starred GitHub repositories using txtai embeddings.

## Overview

GitHub Stars Search is a tool that allows you to search through your starred GitHub repositories using neural embeddings. It extracts README files and repository metadata, processes them into chunks, and creates embeddings for efficient semantic search.

Key features:
- Fetches your starred GitHub repositories and their metadata
- Extracts README files (preferably in English)
- Processes content using intelligent chunking strategies
- Generates embeddings using BAAI/bge-small-en-v1.5 via txtai
- Provides hybrid search combining neural embeddings and BM25 keyword search
- Stores all data and embeddings on disk for reuse
- Supports incremental updates for newly starred repositories
- Configurable search parameters and embedding models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/github_stars.git
cd github_stars
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your GitHub API key:
Create a `.env` file in the project root with your GitHub API key:
```
GITHUB_STARS_KEY=your_github_api_key
```

You can generate a GitHub API key from your [GitHub Developer Settings](https://github.com/settings/tokens).

## Usage

### Updating Repository Data

Before searching, you need to fetch and process your starred repositories:

```bash
python github_stars_search.py update
```

This will:
1. Fetch your starred repositories from GitHub
2. Extract README files and metadata
3. Process content into chunks
4. Generate embeddings
5. Store everything on disk

You can limit the number of repositories to update:

```bash
python github_stars_search.py update --limit 100
```

Or force update all repositories, even if they haven't changed:

```bash
python github_stars_search.py update --force
```

### Searching Repositories

Once you've updated your repository data, you can search through them:

```bash
python github_stars_search.py search "machine learning for time series"
```

You can apply filters to your search:

```bash
python github_stars_search.py search "machine learning" --min-stars 100 --language python
```

And customize the search weights:

```bash
python github_stars_search.py search "machine learning" --neural-weight 0.8 --keyword-weight 0.2
```

### Web Interface

For a more user-friendly experience, you can use the included web interface:

```bash
# Install Flask if you don't have it
pip install flask

# Run the web interface
python examples/web_interface.py
```

This will start a local web server at http://127.0.0.1:5000 where you can:
- Search your repositories with a simple form
- Adjust neural and keyword search weights
- Filter by language and minimum stars
- View nicely formatted search results

### Configuration

You can view and modify the configuration:

```bash
python github_stars_search.py config --show
```

Set a different embedding model:

```bash
python github_stars_search.py config --embedding-model "sentence-transformers/all-MiniLM-L6-v2"
```

Change search weights:

```bash
python github_stars_search.py config --neural-weight 0.6 --keyword-weight 0.4
```

### Information

View information about your data:

```bash
python github_stars_search.py info
```

## Configuration

The `config.yaml` file contains various settings that you can customize:

- GitHub API settings (pagination, retries, timeout)
- Content processing settings (chunk size, strategy)
- Embedding settings (model, device, batch size)
- Search settings (weights, result limits)
- Storage settings (compression, backups)

## How It Works

### Content Processing

The system uses a hybrid chunking strategy:
1. First attempts to chunk by semantic sections (headers)
2. For large sections, applies sliding window chunking with overlap
3. Preserves repository context in each chunk

### Search

The search engine combines two approaches:
1. Neural search using embeddings for semantic understanding
2. BM25 keyword search for traditional relevance

Results are merged with configurable weights to provide the most relevant repositories.

## Testing

The project includes a comprehensive test suite to ensure code quality and reliability.

### Running Tests

First, install the testing dependencies:

```bash
# Using the provided script
./install_test_deps.sh

# Or manually
pip install pytest pytest-cov pytest-mock
```

Then run the tests:

```bash
# Using the provided scripts
./run_tests.sh                # Run all tests
./run_specific_test.sh tests/test_github_client.py  # Run a specific test file
./run_marked_tests.sh unit    # Run tests with a specific marker

# Or using pytest directly
cd github_stars
pytest
pytest --cov=src  # Run with coverage
```

### Test Categories

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

See the [tests README](tests/README.md) for more information.

## License

MIT
