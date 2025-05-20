#!/usr/bin/env python3
"""
Example script demonstrating how to use GitHub Stars Search programmatically.
"""

import os
import sys
from pathlib import Path

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
    """Run the example."""
    print("GitHub Stars Search - Programmatic Usage Example")
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    storage_manager = StorageManager(config.get("storage", {}))
    github_client = GitHubClient(config.get("github", {}))
    content_processor = ContentProcessor(config.get("content", {}))
    embedding_manager = EmbeddingManager(config.get("embeddings", {}), storage_manager)
    search_engine = SearchEngine(config.get("search", {}), embedding_manager, storage_manager)
    
    # Check if we have data
    if not storage_manager.has_data():
        print("\nNo repository data found. Let's fetch some repositories...")
        
        # Fetch a few repositories
        print("\nFetching repositories...")
        repositories = github_client.get_starred_repositories(limit=5)
        
        print(f"\nProcessing {len(repositories)} repositories...")
        for repo in repositories:
            # Extract README
            readme_content = github_client.get_readme(repo["full_name"])
            
            if readme_content:
                # Process content
                chunks = content_processor.process_readme(readme_content, repo)
                
                # Store repository data
                storage_manager.store_repository(repo, readme_content, chunks)
                
                # Generate embeddings
                embedding_manager.generate_embeddings(repo["id"], chunks)
                
                print(f"  ✓ Processed {repo['full_name']}")
            else:
                print(f"  ✗ No README found for {repo['full_name']}")
    else:
        print("\nFound existing repository data. Regenerating embeddings...")
        
        # Get all repositories
        repositories = storage_manager.get_all_repositories()
        print(f"Found {len(repositories)} repositories")
        
        # Process each repository
        for repo_id in repositories:
            # Get repository data
            repo = storage_manager.get_repository(repo_id)
            if not repo:
                print(f"  ✗ No repository data found for ID {repo_id}")
                continue
                
            # Get chunks for this repository
            chunks = storage_manager.get_repository_chunks(repo_id)
            
            if chunks:
                # Generate embeddings
                embedding_manager.generate_embeddings(repo_id, chunks)
                print(f"  ✓ Regenerated embeddings for {repo['full_name']}")
            else:
                print(f"  ✗ No chunks found for {repo['full_name']}")
    
    # Perform a search
    print("\nPerforming search...")
    query = "machine learning"
    results = search_engine.search(query, limit=5)
    
    # Display results
    print(f"\nSearch results for '{query}':")
    if results:
        for i, result in enumerate(results, 1):
            repo = result["repository"]
            print(f"\n{i}. {repo['full_name']} (Score: {result['score']:.2f})")
            print(f"   Description: {repo.get('description', 'No description')}")
            print(f"   Language: {repo.get('language', 'Unknown')}")
            print(f"   Stars: {repo.get('stargazers_count', 0)}")
            print(f"   URL: {repo.get('html_url', '')}")
    else:
        print("  No results found.")
    
    print("\nExample completed!")

if __name__ == "__main__":
    main()
