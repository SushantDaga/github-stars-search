#!/usr/bin/env python3
"""
Example script demonstrating hybrid search functionality.
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
from src.api.github_client import GitHubClient
from src.processor.content_processor import ContentProcessor
from src.embeddings.embedding_manager import EmbeddingManager
from src.search.search_engine import SearchEngine
from src.storage.storage_manager import StorageManager
from src.cli.utils import load_config

def main():
    """Run the hybrid search example."""
    console = Console()
    console.print("[bold]GitHub Stars Search - Hybrid Search Example[/bold]")
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    storage_manager = StorageManager(config.get("storage", {}))
    embedding_manager = EmbeddingManager(config.get("embeddings", {}), storage_manager)
    
    # Check if we have data
    if not storage_manager.has_data():
        console.print("[bold red]No repository data found. Run 'python github_stars_search.py update' first.[/bold red]")
        return
    
    # Create search engine instances with different configurations
    neural_config = {**config.get("search", {}), "neural_weight": 1.0, "keyword_weight": 0.0, "hybrid_enabled": False}
    keyword_config = {**config.get("search", {}), "neural_weight": 0.0, "keyword_weight": 1.0, "hybrid_enabled": True}
    hybrid_config = {**config.get("search", {}), "neural_weight": 0.7, "keyword_weight": 0.3, "hybrid_enabled": True}
    
    neural_search = SearchEngine(neural_config, embedding_manager, storage_manager)
    keyword_search = SearchEngine(keyword_config, embedding_manager, storage_manager)
    hybrid_search = SearchEngine(hybrid_config, embedding_manager, storage_manager)
    
    # Get search query
    query = "machine learning for time series"
    console.print(f"\nSearch query: [bold cyan]{query}[/bold cyan]")
    
    # Perform searches
    with console.status("[bold green]Searching repositories...[/bold green]"):
        neural_results = neural_search.search(query, limit=5)
        keyword_results = keyword_search.search(query, limit=5)
        hybrid_results = hybrid_search.search(query, limit=5)
    
    # Display neural search results
    console.print("\n[bold]Neural Search Results:[/bold]")
    _display_results(console, neural_results)
    
    # Display keyword search results
    console.print("\n[bold]Keyword (BM25) Search Results:[/bold]")
    _display_results(console, keyword_results)
    
    # Display hybrid search results
    console.print("\n[bold]Hybrid Search Results:[/bold]")
    _display_results(console, hybrid_results)
    
    # Compare results
    console.print("\n[bold]Comparison:[/bold]")
    console.print("Neural search is better for understanding semantic meaning and context.")
    console.print("Keyword search is better for exact term matching.")
    console.print("Hybrid search combines the strengths of both approaches.")
    
    # Example of different weight configurations
    console.print("\n[bold]Try different hybrid weights:[/bold]")
    console.print("- Neural heavy (0.9, 0.1): Better for conceptual queries")
    console.print("- Balanced (0.5, 0.5): Good general-purpose configuration")
    console.print("- Keyword heavy (0.1, 0.9): Better for specific technical terms")
    
    console.print("\nExample completed!")

def _display_results(console, results):
    """Display search results in a table."""
    if not results:
        console.print("  No results found.")
        return
    
    table = Table(show_header=True, header_style="bold")
    
    table.add_column("Repository", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Language", style="blue")
    table.add_column("Stars", style="yellow")
    table.add_column("Score", style="magenta")
    
    for result in results:
        repo = result["repository"]
        table.add_row(
            repo["full_name"],
            repo.get("description", "No description")[:50] + "..." if repo.get("description", "") and len(repo.get("description", "")) > 50 else repo.get("description", "No description"),
            repo.get("language", "Unknown"),
            str(repo.get("stargazers_count", 0)),
            f"{result['score']:.2f}"
        )
    
    console.print(table)

if __name__ == "__main__":
    main()
