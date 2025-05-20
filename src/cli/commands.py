"""
CLI commands for GitHub Stars Search.
"""

import json
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.cli.utils import load_config, save_config
from src.api.github_client import GitHubClient
from src.processor.content_processor import ContentProcessor
from src.embeddings.embedding_manager import EmbeddingManager
from src.search.search_engine import SearchEngine
from src.storage.storage_manager import StorageManager

# Initialize console for rich output
console = Console()
logger = logging.getLogger(__name__)

def search_command(search_engine, query, neural_weight=None, keyword_weight=None, min_stars=None, language=None, limit=None, json_output=False):
    """
    Search for repositories matching the query.
    
    Args:
        search_engine: The search engine instance
        query (str): Search query
        neural_weight (float, optional): Weight for neural search results
        keyword_weight (float, optional): Weight for keyword search results
        min_stars (int, optional): Minimum number of stars
        language (str, optional): Filter by programming language
        limit (int, optional): Maximum number of results to return
        json_output (bool, optional): Output results as JSON
        
    Returns:
        bool: True if successful
    """
    # Update search engine weights if provided
    if neural_weight is not None:
        search_engine.neural_weight = neural_weight
    if keyword_weight is not None:
        search_engine.keyword_weight = keyword_weight
    
    # Prepare filters
    filters = {}
    if min_stars is not None:
        filters["stargazers_count"] = {"min": min_stars}
    if language is not None:
        filters["language"] = language
    
    # Execute search
    with console.status("[bold green]Searching repositories...[/bold green]"):
        results = search_engine.search(query, filters=filters, limit=limit)
    
    # Output results
    if json_output:
        console.print(json.dumps(results, indent=2))
    else:
        _display_search_results(results)
    
    return True

def update_command(github_client, content_processor, storage_manager, embedding_manager, force=False, limit=None):
    """
    Update repository data and embeddings.
    
    Args:
        github_client: The GitHub client instance
        content_processor: The content processor instance
        storage_manager: The storage manager instance
        embedding_manager: The embedding manager instance
        force (bool, optional): Force update all repositories
        limit (int, optional): Limit the number of repositories to update
        
    Returns:
        bool: True if successful
    """
    
    # Fetch starred repositories
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}[/bold green]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Fetching starred repositories...", total=None)
        
        # Get repositories
        repositories = github_client.get_starred_repositories(limit=limit)
        
        # Update progress with total count
        total_repos = len(repositories)
        progress.update(task, total=total_repos, completed=0)
        progress.update(task, description=f"Processing {total_repos} repositories...")
        
        # Process repositories
        for i, repo in enumerate(repositories):
            repo_id = repo["id"]
            
            # Skip if not forced and already processed
            if not force and storage_manager.has_repository(repo_id) and not storage_manager.is_repository_outdated(repo):
                progress.update(task, advance=1)
                continue
            
            # Process repository content
            try:
                # Extract README and other content
                readme_content = github_client.get_readme(repo["full_name"])
                if readme_content:
                    # Process content
                    chunks = content_processor.process_readme(readme_content, repo)
                    
                    # Store repository data
                    storage_manager.store_repository(repo, readme_content, chunks)
                    
                    # Generate embeddings
                    embedding_manager.generate_embeddings(repo_id, chunks)
                else:
                    logger.warning(f"No README found for {repo['full_name']}")
            except Exception as e:
                logger.error(f"Error processing repository {repo['full_name']}: {str(e)}")
            
            # Update progress
            progress.update(task, advance=1)
    
    console.print(f"[bold green]Successfully processed {total_repos} repositories.[/bold green]")
    return True

def config_command(embedding_model=None, device=None, neural_weight=None, keyword_weight=None, chunk_strategy=None, chunk_size=None, chunk_overlap=None, show=False):
    """
    Configure search settings.
    
    Args:
        embedding_model (str, optional): Set the embedding model
        device (str, optional): Set the device for embeddings (cpu/cuda)
        neural_weight (float, optional): Set weight for neural search
        keyword_weight (float, optional): Set weight for keyword search
        chunk_strategy (str, optional): Set the chunk strategy (hybrid/semantic)
        chunk_size (int, optional): Set the maximum chunk size
        chunk_overlap (int, optional): Set the chunk overlap
        show (bool, optional): Show current configuration
        
    Returns:
        bool: True if successful
    """
    config = load_config()
    
    # Show current configuration
    if show:
        _display_configuration(config)
        return True
    
    # Update configuration
    if embedding_model is not None:
        if "embeddings" not in config:
            config["embeddings"] = {}
        config["embeddings"]["model"] = embedding_model
    
    if device is not None:
        if "embeddings" not in config:
            config["embeddings"] = {}
        config["embeddings"]["device"] = device
    
    if neural_weight is not None:
        if "search" not in config:
            config["search"] = {}
        config["search"]["neural_weight"] = neural_weight
    
    if keyword_weight is not None:
        if "search" not in config:
            config["search"] = {}
        config["search"]["keyword_weight"] = keyword_weight
    
    if chunk_strategy is not None:
        if "content" not in config:
            config["content"] = {}
        config["content"]["chunk_strategy"] = chunk_strategy
    
    if chunk_size is not None:
        if "content" not in config:
            config["content"] = {}
        config["content"]["max_chunk_size"] = chunk_size
    
    if chunk_overlap is not None:
        if "content" not in config:
            config["content"] = {}
        config["content"]["chunk_overlap"] = chunk_overlap
    
    # Save configuration
    save_config(config)
    
    # Show updated configuration
    _display_configuration(config)
    
    return True

def info_command(storage_manager, embedding_manager):
    """
    Show information about the current data and embeddings.
    
    Args:
        storage_manager: The storage manager instance
        embedding_manager: The embedding manager instance
        
    Returns:
        bool: True if successful
    """
    # Get repository stats
    repo_count = storage_manager.get_repository_count()
    embedding_count = storage_manager.get_embedding_count()
    
    # Get all repositories
    repositories = storage_manager.get_all_repositories()
    
    # Get model info
    model_info = embedding_manager.get_model_info()
    
    # Create table
    table = Table(title="GitHub Stars Search - Information")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Display repository information
    if repo_count > 0:
        table.add_row("Repositories", str(repo_count))
        table.add_row("Embedded Chunks", str(embedding_count))
        
        # Display language distribution
        languages = {}
        for repo in repositories.values():
            lang = repo.get("language")
            if lang:
                languages[lang] = languages.get(lang, 0) + 1
        
        if languages:
            table.add_row("Languages", ", ".join(f"{lang} ({count})" for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]))
    else:
        table.add_row("Repositories", "[yellow]No repositories found.[/yellow]")
    
    # Display model information
    if model_info:
        table.add_row("Embedding Model", model_info.get("model_name", "Not set"))
        table.add_row("Device", model_info.get("device", "cpu"))
    
    # Display table
    console.print(table)
    
    return True

def _display_search_results(results):
    """
    Display search results in a table.
    
    Args:
        results (list): Search results
    """
    if not results:
        console.print("[bold yellow]No results found.[/bold yellow]")
        return
    
    table = Table(title="Search Results")
    
    table.add_column("Repository", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Language", style="blue")
    table.add_column("Stars", style="yellow")
    table.add_column("Score", style="magenta")
    
    for result in results:
        repo = result["repository"]
        table.add_row(
            repo["full_name"],
            repo.get("description", "No description"),
            repo.get("language", "Unknown"),
            str(repo.get("stargazers_count", 0)),
            f"{result['score']:.2f}"
        )
    
    console.print(table)

def _display_configuration(config):
    """
    Display current configuration.
    
    Args:
        config (dict): Configuration dictionary
    """
    table = Table(title="Current Configuration")
    
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # Embedding settings
    embedding_config = config.get("embeddings", {})
    table.add_row("Embedding Model", embedding_config.get("model", "Default"))
    table.add_row("Device", embedding_config.get("device", "cpu"))
    
    # Search settings
    search_config = config.get("search", {})
    table.add_row("Hybrid Search", "Enabled" if search_config.get("hybrid_enabled", True) else "Disabled")
    table.add_row("Neural Weight", str(search_config.get("neural_weight", 0.7)))
    table.add_row("Keyword Weight", str(search_config.get("keyword_weight", 0.3)))
    table.add_row("Max Results", str(search_config.get("max_results", 20)))
    
    console.print(table)
