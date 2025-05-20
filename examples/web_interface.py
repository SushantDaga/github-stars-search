#!/usr/bin/env python3
"""
Example web interface for GitHub Stars Search using Flask.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
from src.api.github_client import GitHubClient
from src.processor.content_processor import ContentProcessor
from src.embeddings.embedding_manager import EmbeddingManager
from src.search.search_engine import SearchEngine
from src.storage.storage_manager import StorageManager
from src.cli.utils import load_config

# Initialize Flask app
app = Flask(__name__)

# Initialize components
config = load_config()
storage_manager = StorageManager(config.get("storage", {}))
embedding_manager = EmbeddingManager(config.get("embeddings", {}), storage_manager)
search_engine = SearchEngine(config.get("search", {}), embedding_manager, storage_manager)

# Create HTML template directory
template_dir = Path(__file__).parent / "templates"
template_dir.mkdir(exist_ok=True)

# Create HTML template
template_path = template_dir / "index.html"
if not template_path.exists():
    with open(template_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>GitHub Stars Search</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #0366d6;
            margin-bottom: 20px;
        }
        .search-container {
            margin-bottom: 30px;
        }
        .search-box {
            padding: 10px;
            width: 70%;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .search-button {
            padding: 10px 20px;
            background-color: #0366d6;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .search-button:hover {
            background-color: #0353b4;
        }
        .filters {
            margin: 20px 0;
            padding: 15px;
            background-color: #f6f8fa;
            border-radius: 4px;
        }
        .filters label {
            margin-right: 10px;
        }
        .filters input, .filters select {
            margin-right: 20px;
            padding: 5px;
        }
        .result {
            margin-bottom: 30px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .result h2 {
            margin-top: 0;
            color: #0366d6;
        }
        .result h2 a {
            text-decoration: none;
            color: #0366d6;
        }
        .result h2 a:hover {
            text-decoration: underline;
        }
        .result-meta {
            color: #586069;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .result-meta span {
            margin-right: 15px;
        }
        .result-description {
            margin-bottom: 10px;
        }
        .result-score {
            font-weight: bold;
            color: #0366d6;
        }
        .no-results {
            padding: 20px;
            background-color: #f6f8fa;
            border-radius: 4px;
            text-align: center;
            color: #586069;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .search-weights {
            display: flex;
            align-items: center;
            margin-top: 10px;
        }
        .search-weights label {
            margin-right: 10px;
        }
        .search-weights input {
            width: 60px;
            margin-right: 20px;
        }
    </style>
</head>
<body>
    <h1>GitHub Stars Search</h1>
    
    <div class="search-container">
        <form id="search-form" action="/search" method="get">
            <input type="text" name="query" id="query" class="search-box" placeholder="Search your starred repositories..." required>
            <button type="submit" class="search-button">Search</button>
            
            <div class="search-weights">
                <label for="neural-weight">Neural Weight:</label>
                <input type="number" id="neural-weight" name="neural_weight" min="0" max="1" step="0.1" value="0.7">
                
                <label for="keyword-weight">Keyword Weight:</label>
                <input type="number" id="keyword-weight" name="keyword_weight" min="0" max="1" step="0.1" value="0.3">
            </div>
            
            <div class="filters">
                <label for="min-stars">Min Stars:</label>
                <input type="number" id="min-stars" name="min_stars" min="0">
                
                <label for="language">Language:</label>
                <input type="text" id="language" name="language" placeholder="e.g. Python">
                
                <label for="limit">Results:</label>
                <select id="limit" name="limit">
                    <option value="10">10</option>
                    <option value="20" selected>20</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                </select>
            </div>
        </form>
    </div>
    
    <div id="loading" class="loading">
        <p>Searching repositories...</p>
    </div>
    
    <div id="results">
        {% if results %}
            {% for result in results %}
                <div class="result">
                    <h2><a href="{{ result.repository.html_url }}" target="_blank">{{ result.repository.full_name }}</a></h2>
                    <div class="result-meta">
                        <span>Language: {{ result.repository.language or 'Unknown' }}</span>
                        <span>Stars: {{ result.repository.stargazers_count }}</span>
                        <span>Forks: {{ result.repository.forks_count }}</span>
                        <span class="result-score">Score: {{ "%.2f"|format(result.score) }}</span>
                    </div>
                    <div class="result-description">
                        {{ result.repository.description or 'No description available.' }}
                    </div>
                </div>
            {% endfor %}
        {% elif query %}
            <div class="no-results">
                <p>No results found for "{{ query }}".</p>
            </div>
        {% endif %}
    </div>
    
    <script>
        document.getElementById('search-form').addEventListener('submit', function() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
        });
    </script>
</body>
</html>""")

@app.route('/')
def index():
    """Render the search page."""
    return render_template('index.html')

@app.route('/search')
def search():
    """Handle search requests."""
    query = request.args.get('query', '')
    neural_weight = float(request.args.get('neural_weight', 0.7))
    keyword_weight = float(request.args.get('keyword_weight', 0.3))
    min_stars = request.args.get('min_stars', '')
    language = request.args.get('language', '')
    limit = int(request.args.get('limit', 20))
    
    # Update search engine configuration
    search_engine.neural_weight = neural_weight
    search_engine.keyword_weight = keyword_weight
    
    # Prepare filters
    filters = {}
    if min_stars and min_stars.isdigit():
        filters["stargazers_count"] = {"min": int(min_stars)}
    if language:
        filters["language"] = language
    
    # Perform search
    results = []
    if query:
        results = search_engine.search(query, filters=filters, limit=limit)
    
    return render_template('index.html', results=results, query=query)

@app.route('/api/search')
def api_search():
    """API endpoint for search."""
    query = request.args.get('query', '')
    neural_weight = float(request.args.get('neural_weight', 0.7))
    keyword_weight = float(request.args.get('keyword_weight', 0.3))
    min_stars = request.args.get('min_stars', '')
    language = request.args.get('language', '')
    limit = int(request.args.get('limit', 20))
    
    # Update search engine configuration
    search_engine.neural_weight = neural_weight
    search_engine.keyword_weight = keyword_weight
    
    # Prepare filters
    filters = {}
    if min_stars and min_stars.isdigit():
        filters["stargazers_count"] = {"min": int(min_stars)}
    if language:
        filters["language"] = language
    
    # Perform search
    results = []
    if query:
        results = search_engine.search(query, filters=filters, limit=limit)
    
    # Convert results to JSON-serializable format
    serialized_results = []
    for result in results:
        serialized_results.append({
            "id": result["id"],
            "score": result["score"],
            "repository": {
                "id": result["repository"]["id"],
                "full_name": result["repository"]["full_name"],
                "description": result["repository"].get("description"),
                "html_url": result["repository"].get("html_url"),
                "language": result["repository"].get("language"),
                "stargazers_count": result["repository"].get("stargazers_count"),
                "forks_count": result["repository"].get("forks_count"),
                "watchers_count": result["repository"].get("watchers_count")
            }
        })
    
    return jsonify({"results": serialized_results})

def main():
    """Run the Flask application."""
    # Check if we have data
    if not storage_manager.has_data():
        print("No repository data found. Run 'python github_stars_search.py update' first.")
        return
    
    # Run the Flask app
    print("Starting web interface at http://127.0.0.1:5000")
    app.run(debug=True)

if __name__ == "__main__":
    main()
