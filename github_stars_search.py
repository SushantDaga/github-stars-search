#!/usr/bin/env python3
"""
GitHub Stars Search - Neural search for your starred GitHub repositories.

This tool fetches your starred GitHub repositories, extracts README files and metadata,
and provides a neural search interface using txtai.
"""

import os
import sys
import click
from rich.console import Console
from rich.progress import Progress

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import project modules
from src.cli.commands import search_command, update_command, config_command, info_command
from src.cli.utils import load_config, setup_logging, check_environment, get_components

# Initialize console for rich output
console = Console()

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """GitHub Stars Search - Neural search for your starred GitHub repositories."""
    # Check environment and setup
    try:
        check_environment()
        setup_logging()
    except Exception as e:
        console.print(f"[bold red]Error during initialization:[/bold red] {str(e)}")
        sys.exit(1)

@cli.command()
@click.argument("query", required=True)
@click.option("--neural-weight", type=float, help="Weight for neural search results (0.0-1.0)")
@click.option("--keyword-weight", type=float, help="Weight for keyword search results (0.0-1.0)")
@click.option("--min-stars", type=int, help="Minimum number of stars")
@click.option("--language", help="Filter by programming language")
@click.option("--limit", type=int, help="Maximum number of results to return")
@click.option("--json", is_flag=True, help="Output results as JSON")
def search(query, neural_weight, keyword_weight, min_stars, language, limit, json):
    """Search your starred GitHub repositories."""
    try:
        # Get components
        components = get_components()
        
        # Call the search command
        search_command(
            search_engine=components["search_engine"],
            query=query,
            neural_weight=neural_weight,
            keyword_weight=keyword_weight,
            min_stars=min_stars,
            language=language,
            limit=limit,
            json_output=json
        )
    except Exception as e:
        console.print(f"[bold red]Search error:[/bold red] {str(e)}")
        sys.exit(1)

@cli.command()
@click.option("--force", is_flag=True, help="Force update all repositories")
@click.option("--limit", type=int, help="Limit the number of repositories to update")
def update(force, limit):
    """Update repository data and embeddings."""
    try:
        # Get components
        components = get_components()
        
        # Call the update command
        update_command(
            github_client=components["github_client"],
            content_processor=components["content_processor"],
            storage_manager=components["storage_manager"],
            embedding_manager=components["embedding_manager"],
            force=force,
            limit=limit
        )
    except Exception as e:
        console.print(f"[bold red]Update error:[/bold red] {str(e)}")
        sys.exit(1)

@cli.command()
@click.option("--embedding-model", help="Set the embedding model")
@click.option("--device", help="Set the device for embeddings (cpu/cuda)")
@click.option("--neural-weight", type=float, help="Set weight for neural search")
@click.option("--keyword-weight", type=float, help="Set weight for keyword search")
@click.option("--chunk-strategy", help="Set the chunk strategy (hybrid/semantic)")
@click.option("--chunk-size", type=int, help="Set the maximum chunk size")
@click.option("--chunk-overlap", type=int, help="Set the chunk overlap")
@click.option("--show", is_flag=True, help="Show current configuration")
def config(embedding_model, device, neural_weight, keyword_weight, chunk_strategy, chunk_size, chunk_overlap, show):
    """Configure search settings."""
    try:
        config_command(
            embedding_model=embedding_model,
            device=device,
            neural_weight=neural_weight,
            keyword_weight=keyword_weight,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            show=show
        )
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/bold red] {str(e)}")
        sys.exit(1)

@cli.command()
def info():
    """Show information about the current data and embeddings."""
    try:
        # Get components
        components = get_components()
        
        # Call the info command
        info_command(
            storage_manager=components["storage_manager"],
            embedding_manager=components["embedding_manager"]
        )
    except Exception as e:
        console.print(f"[bold red]Info error:[/bold red] {str(e)}")
        sys.exit(1)

def main():
    """Main entry point for the application."""
    try:
        cli()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
