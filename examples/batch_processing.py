#!/usr/bin/env python3
"""
Example script demonstrating batch processing of repositories.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
from src.api.github_client import GitHubClient
from src.processor.content_processor import ContentProcessor
from src.embeddings.embedding_manager import EmbeddingManager
from src.search.search_engine import SearchEngine
from src.storage.storage_manager import StorageManager
from src.cli.utils import load_config

def process_repository(repo, github_client, content_processor, storage_manager, embedding_manager):
    """
    Process a single repository.
    
    Args:
        repo (dict): Repository data
        github_client (GitHubClient): GitHub client
        content_processor (ContentProcessor): Content processor
        storage_manager (StorageManager): Storage manager
        embedding_manager (EmbeddingManager): Embedding manager
    
    Returns:
        tuple: (repo_id, success)
    """
    repo_id = repo["id"]
    
    try:
        # Skip if already processed and not outdated
        if storage_manager.has_repository(repo_id) and not storage_manager.is_repository_outdated(repo):
            return repo_id, "skipped"
        
        # Extract README
        readme_content = github_client.get_readme(repo["full_name"])
        
        if not readme_content:
            return repo_id, "no_readme"
        
        # Process content
        chunks = content_processor.process_readme(readme_content, repo)
        
        # Store repository data
        storage_manager.store_repository(repo, readme_content, chunks)
        
        # Generate embeddings
        embedding_manager.generate_embeddings(repo_id, chunks)
        
        return repo_id, "success"
    
    except Exception as e:
        return repo_id, f"error: {str(e)}"

def main():
    """Run the batch processing example."""
    parser = argparse.ArgumentParser(description="Batch process GitHub repositories")
    parser.add_argument("--limit", type=int, help="Limit the number of repositories to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--output", type=str, help="Output file for processing results")
    parser.add_argument("--force", action="store_true", help="Force processing of all repositories")
    args = parser.parse_args()
    
    print("GitHub Stars Search - Batch Processing Example")
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    storage_manager = StorageManager(config.get("storage", {}))
    github_client = GitHubClient(config.get("github", {}))
    content_processor = ContentProcessor(config.get("content", {}))
    embedding_manager = EmbeddingManager(config.get("embeddings", {}), storage_manager)
    
    # Fetch repositories
    print("\nFetching repositories...")
    repositories = github_client.get_starred_repositories(limit=args.limit)
    
    print(f"\nProcessing {len(repositories)} repositories with {args.workers} workers...")
    
    # Process repositories in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit tasks
        futures = {
            executor.submit(
                process_repository, 
                repo, 
                github_client, 
                content_processor, 
                storage_manager, 
                embedding_manager
            ): repo["full_name"] for repo in repositories
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            repo_name = futures[future]
            try:
                repo_id, status = future.result()
                results[repo_name] = status
            except Exception as e:
                results[repo_name] = f"error: {str(e)}"
    
    # Summarize results
    success_count = sum(1 for status in results.values() if status == "success")
    skipped_count = sum(1 for status in results.values() if status == "skipped")
    no_readme_count = sum(1 for status in results.values() if status == "no_readme")
    error_count = sum(1 for status in results.values() if status.startswith("error"))
    
    print("\nProcessing complete!")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  No README: {no_readme_count}")
    print(f"  Errors: {error_count}")
    
    # Save results to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Perform a test search
    print("\nPerforming a test search...")
    search_engine = SearchEngine(config.get("search", {}), embedding_manager, storage_manager)
    query = "machine learning"
    search_results = search_engine.search(query, limit=5)
    
    print(f"\nSearch results for '{query}':")
    if search_results:
        for i, result in enumerate(search_results, 1):
            repo = result["repository"]
            print(f"\n{i}. {repo['full_name']} (Score: {result['score']:.2f})")
            print(f"   Description: {repo.get('description', 'No description')}")
            print(f"   Language: {repo.get('language', 'Unknown')}")
            print(f"   Stars: {repo.get('stargazers_count', 0)}")
    else:
        print("  No results found.")
    
    print("\nExample completed!")

if __name__ == "__main__":
    main()
