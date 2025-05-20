# GitHub Stars Search Examples

This directory contains example scripts demonstrating how to use GitHub Stars Search.

## Examples

### 1. Programmatic Usage

[programmatic_usage.py](programmatic_usage.py) demonstrates how to use the GitHub Stars Search components programmatically in your own Python code.

```bash
python programmatic_usage.py
```

This example shows:
- How to initialize the various components
- How to fetch and process repositories
- How to perform searches
- How to access and display search results

### 2. Hybrid Search Example

[hybrid_search_example.py](hybrid_search_example.py) demonstrates the hybrid search functionality, comparing neural, keyword, and hybrid search approaches.

```bash
python hybrid_search_example.py
```

This example shows:
- How to configure different search engines with different weights
- How to perform neural, keyword, and hybrid searches
- How to compare the results from different search approaches
- How different weight configurations affect search results

### 3. Web Interface

[web_interface.py](web_interface.py) provides a simple web interface for searching your starred repositories using Flask.

```bash
# Install Flask if you don't have it
pip install flask

# Run the web interface
python web_interface.py
```

This example shows:
- How to create a web interface for the search functionality
- How to handle search parameters from web forms
- How to display search results in a web page
- How to create a simple REST API endpoint for search

### 4. Batch Processing

[batch_processing.py](batch_processing.py) demonstrates how to process multiple repositories in parallel using multithreading.

```bash
# Process all repositories with 4 worker threads
python batch_processing.py --workers 4

# Process a limited number of repositories
python batch_processing.py --limit 100 --workers 8

# Save processing results to a file
python batch_processing.py --output results.json
```

This example shows:
- How to process repositories in parallel using ThreadPoolExecutor
- How to track and report processing progress
- How to handle errors and summarize results
- How to customize processing with command-line arguments

### 5. Custom Embedding Models

[custom_embedding_model.py](custom_embedding_model.py) demonstrates how to use different embedding models for search.

```bash
# Use the default model (bge-small)
python custom_embedding_model.py

# Try a different model
python custom_embedding_model.py --model minilm

# Compare with a custom query
python custom_embedding_model.py --model mpnet --query "natural language processing"

# Update the config.yaml with the selected model
python custom_embedding_model.py --model paraphrase --update-config
```

This example shows:
- How to use different embedding models with the search engine
- How to compare search results between different models
- How to update the configuration with a new model
- The strengths and weaknesses of different embedding models

### 6. Export Search Results

[export_results.py](export_results.py) demonstrates how to export search results to various formats.

```bash
# Export to all supported formats
python export_results.py

# Export to a specific format
python export_results.py --format json

# Customize the search query and limit
python export_results.py --query "deep learning" --limit 20

# Specify a custom output directory
python export_results.py --output-dir "./my_exports"
```

This example shows:
- How to export search results to JSON, CSV, Markdown, HTML, and XML formats
- How to customize the search query and result limit
- How to format and structure data for different output formats
- How to create well-formatted reports from search results

## Creating Your Own Examples

You can use these examples as a starting point for creating your own custom applications using GitHub Stars Search. The key components you'll need to work with are:

1. `GitHubClient` - For fetching repository data from GitHub
2. `ContentProcessor` - For processing and chunking README content
3. `EmbeddingManager` - For generating and managing embeddings
4. `SearchEngine` - For performing searches
5. `StorageManager` - For storing and retrieving repository data and embeddings

See the [main README](../README.md) for more information on the overall architecture and configuration options.
