"""
CLI utilities for GitHub Stars Search.
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from rich.console import Console
from dotenv import load_dotenv

# Initialize console for rich output
console = Console()

def load_config():
    """
    Load configuration from config.yaml.
    
    Returns:
        dict: Configuration dictionary
    """
    # Default configuration
    default_config = {
        "github": {
            "per_page": 100,
            "max_retries": 3,
            "timeout": 30
        },
        "content": {
            "max_readme_size": 500000,
            "chunk_strategy": "hybrid",
            "max_chunk_size": 512,
            "chunk_overlap": 50
        },
        "embeddings": {
            "model": "BAAI/bge-small-en-v1.5",
            "device": "cpu",
            "batch_size": 32,
            "cache_enabled": True
        },
        "search": {
            "hybrid_enabled": True,
            "neural_weight": 0.7,
            "keyword_weight": 0.3,
            "max_results": 20,
            "min_score": 0.2
        },
        "storage": {
            "compress_data": True,
            "backup_enabled": True,
            "max_backups": 3
        }
    }
    
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    if not config_path.exists():
        console.print("[bold yellow]Warning:[/bold yellow] config.yaml not found, using default configuration.")
        return default_config
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if config is None:
            console.print("[bold yellow]Warning:[/bold yellow] Empty config file, using default configuration.")
            return default_config
            
        return config
    except Exception as e:
        console.print(f"[bold red]Error loading configuration:[/bold red] {str(e)}")
        return default_config

def save_config(config):
    """
    Save configuration to config.yaml.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        console.print("[bold green]Configuration saved successfully.[/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]Error saving configuration:[/bold red] {str(e)}")
        return False

def setup_logging():
    """
    Set up logging configuration.
    """
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "github_stars_search.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_environment():
    """
    Check if the environment is properly set up.
    
    Raises:
        Exception: If required environment variables are missing
    """
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    # Check for GitHub API key
    if not os.environ.get("GITHUB_STARS_KEY"):
        raise Exception("GITHUB_STARS_KEY environment variable not found. Please set it in .env file.")
    
    # Check for data directories
    data_dir = Path(__file__).parent.parent.parent / "data"
    for subdir in ["repositories", "embeddings", "index"]:
        (data_dir / subdir).mkdir(exist_ok=True)

def get_github_api_key():
    """
    Get the GitHub API key from environment variables or .env file.
    
    Returns:
        str: GitHub API key
    
    Raises:
        ValueError: If no API key is found
    """
    # Check environment variables
    api_key = os.environ.get("GITHUB_STARS_KEY")
    if api_key:
        return api_key
    
    # Try to load from .env file
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("GITHUB_STARS_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        if api_key:
                            return api_key
        except Exception:
            pass
    
    # Ask the user for the API key
    api_key = input("Enter your GitHub API key: ")
    if not api_key:
        raise ValueError("GitHub API key is required.")
    
    return api_key

def get_components():
    """
    Initialize and return all components needed for the application.
    
    Returns:
        dict: Dictionary containing all initialized components
    """
    # Import components here to avoid circular imports
    from src.api.github_client import GitHubClient
    from src.processor.content_processor import ContentProcessor
    from src.storage.storage_manager import StorageManager
    from src.embeddings.embedding_manager import EmbeddingManager
    from src.search.search_engine import SearchEngine
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    storage_manager = StorageManager(config.get("storage", {}))
    github_client = GitHubClient(config.get("github", {}))
    content_processor = ContentProcessor(config.get("content", {}))
    embedding_manager = EmbeddingManager(config.get("embeddings", {}), storage_manager)
    search_engine = SearchEngine(config.get("search", {}), embedding_manager, storage_manager)
    
    # Return components as a dictionary
    return {
        "github_client": github_client,
        "content_processor": content_processor,
        "storage_manager": storage_manager,
        "embedding_manager": embedding_manager,
        "search_engine": search_engine
    }
