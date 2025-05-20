#!/usr/bin/env python3
"""
Example script demonstrating how to use a custom embedding model.
"""

import os
import sys
import argparse
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
from src.cli.utils import load_config, save_config

# Available embedding models
EMBEDDING_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "distilbert": "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    "paraphrase": "sentence-transformers/paraphrase-MiniLM-L3-v2"
}

def main():
    """Run the custom embedding model example."""
    parser = argparse.ArgumentParser(description="Use a custom embedding model")
    parser.add_argument("--model", type=str, choices=list(EMBEDDING_MODELS.keys()), default="bge-small",
                        help="Embedding model to use")
    parser.add_argument("--query", type=str, default="machine learning",
                        help="Search query")
    parser.add_argument("--limit", type=int, default=5,
                        help="Maximum number of results to return")
    parser.add_argument("--update-config", action="store_true",
                        help="Update the config.yaml file with the selected model")
    args = parser.parse_args()
    
    console = Console()
    console.print(f"[bold]GitHub Stars Search - Custom Embedding Model Example[/bold]")
    
    # Get the selected model
    model_name = EMBEDDING_MODELS[args.model]
    console.print(f"\nUsing embedding model: [bold cyan]{model_name}[/bold cyan]")
    
    # Load configuration
    config = load_config()
    
    # Update configuration with selected model
    if "embeddings" not in config:
        config["embeddings"] = {}
    
    original_model = config["embeddings"].get("model")
    config["embeddings"]["model"] = model_name
    
    # Save configuration if requested
    if args.update_config:
        save_config(config)
        console.print(f"[bold green]Updated config.yaml with model: {model_name}[/bold green]")
    
    # Initialize components
    storage_manager = StorageManager(config.get("storage", {}))
    
    # Check if we have data
    if not storage_manager.has_data():
        console.print("[bold red]No repository data found. Run 'python github_stars_search.py update' first.[/bold red]")
        return
    
    # Initialize embedding manager with custom model
    embedding_manager = EmbeddingManager(config.get("embeddings", {}), storage_manager)
    
    # Initialize search engine
    search_engine = SearchEngine(config.get("search", {}), embedding_manager, storage_manager)
    
    # Perform search
    console.print(f"\nSearching for: [bold]{args.query}[/bold]")
    with console.status("[bold green]Searching repositories...[/bold green]"):
        results = search_engine.search(args.query, limit=args.limit)
    
    # Display results
    console.print(f"\n[bold]Search Results with {args.model} model:[/bold]")
    if results:
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
                (repo.get("description", "No description")[:50] + "...") if repo.get("description", "") and len(repo.get("description", "")) > 50 else repo.get("description", "No description"),
                repo.get("language", "Unknown"),
                str(repo.get("stargazers_count", 0)),
                f"{result['score']:.2f}"
            )
        
        console.print(table)
    else:
        console.print("  No results found.")
    
    # Compare with different models
    if args.model != "bge-small":
        console.print("\n[bold]Let's compare with the default model (bge-small):[/bold]")
        
        # Restore original model
        config["embeddings"]["model"] = "BAAI/bge-small-en-v1.5"
        
        # Initialize with default model
        default_embedding_manager = EmbeddingManager(config.get("embeddings", {}), storage_manager)
        default_search_engine = SearchEngine(config.get("search", {}), default_embedding_manager, storage_manager)
        
        # Perform search with default model
        with console.status("[bold green]Searching with default model...[/bold green]"):
            default_results = default_search_engine.search(args.query, limit=args.limit)
        
        # Display results
        console.print(f"\n[bold]Search Results with default model (bge-small):[/bold]")
        if default_results:
            table = Table(show_header=True, header_style="bold")
            
            table.add_column("Repository", style="cyan")
            table.add_column("Description", style="green")
            table.add_column("Language", style="blue")
            table.add_column("Stars", style="yellow")
            table.add_column("Score", style="magenta")
            
            for result in default_results:
                repo = result["repository"]
                table.add_row(
                    repo["full_name"],
                    (repo.get("description", "No description")[:50] + "...") if repo.get("description", "") and len(repo.get("description", "")) > 50 else repo.get("description", "No description"),
                    repo.get("language", "Unknown"),
                    str(repo.get("stargazers_count", 0)),
                    f"{result['score']:.2f}"
                )
            
            console.print(table)
        else:
            console.print("  No results found.")
    
    # Restore original model if not updating config
    if not args.update_config and original_model:
        config["embeddings"]["model"] = original_model
        save_config(config)
    
    console.print("\n[bold]Model Comparison:[/bold]")
    console.print("Different embedding models have different strengths and weaknesses:")
    console.print("- [bold]bge-small[/bold]: Good balance of performance and size, works well for both symmetric and asymmetric search")
    console.print("- [bold]minilm[/bold]: Smaller and faster, good for general-purpose semantic search")
    console.print("- [bold]mpnet[/bold]: Higher quality but larger and slower")
    console.print("- [bold]distilbert[/bold]: Good for sentence similarity tasks")
    console.print("- [bold]paraphrase[/bold]: Optimized for paraphrase detection, very small and fast")
    
    console.print("\nExample completed!")

if __name__ == "__main__":
    main()
