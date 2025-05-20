#!/usr/bin/env python3
"""
Example script demonstrating how to export search results to different formats.
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from rich.console import Console

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
from src.api.github_client import GitHubClient
from src.processor.content_processor import ContentProcessor
from src.embeddings.embedding_manager import EmbeddingManager
from src.search.search_engine import SearchEngine
from src.storage.storage_manager import StorageManager
from src.cli.utils import load_config

def export_to_json(results, output_file):
    """
    Export search results to JSON format.
    
    Args:
        results (list): Search results
        output_file (str): Output file path
    """
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        serializable_results.append({
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
                "watchers_count": result["repository"].get("watchers_count"),
                "created_at": result["repository"].get("created_at"),
                "updated_at": result["repository"].get("updated_at")
            },
            "chunk_type": result.get("chunk_type"),
            "text_snippet": result.get("text", "")[:200] + "..." if result.get("text") and len(result.get("text", "")) > 200 else result.get("text", "")
        })
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"results": serializable_results, "timestamp": datetime.now().isoformat()}, f, indent=2)

def export_to_csv(results, output_file):
    """
    Export search results to CSV format.
    
    Args:
        results (list): Search results
        output_file (str): Output file path
    """
    # Define CSV fields
    fields = [
        "id", "score", "repository_id", "repository_name", "description", 
        "language", "stars", "forks", "url", "chunk_type"
    ]
    
    # Write to file
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for result in results:
            repo = result["repository"]
            writer.writerow({
                "id": result["id"],
                "score": result["score"],
                "repository_id": repo["id"],
                "repository_name": repo["full_name"],
                "description": repo.get("description", ""),
                "language": repo.get("language", ""),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "url": repo.get("html_url", ""),
                "chunk_type": result.get("chunk_type", "")
            })

def export_to_markdown(results, output_file):
    """
    Export search results to Markdown format.
    
    Args:
        results (list): Search results
        output_file (str): Output file path
    """
    # Generate markdown content
    markdown_content = f"# Search Results\n\n"
    markdown_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, result in enumerate(results, 1):
        repo = result["repository"]
        
        markdown_content += f"## {i}. [{repo['full_name']}]({repo.get('html_url', '')})\n\n"
        markdown_content += f"**Score:** {result['score']:.2f}\n\n"
        
        if repo.get("description"):
            markdown_content += f"**Description:** {repo.get('description')}\n\n"
        
        markdown_content += f"**Language:** {repo.get('language', 'Unknown')}  \n"
        markdown_content += f"**Stars:** {repo.get('stargazers_count', 0)}  \n"
        markdown_content += f"**Forks:** {repo.get('forks_count', 0)}  \n\n"
        
        if result.get("text"):
            markdown_content += "**Matching Content:**\n\n"
            markdown_content += f"```\n{result.get('text', '')[:300]}...\n```\n\n"
        
        markdown_content += "---\n\n"
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

def export_to_html(results, output_file):
    """
    Export search results to HTML format.
    
    Args:
        results (list): Search results
        output_file (str): Output file path
    """
    # Generate HTML content
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>GitHub Stars Search Results</title>
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
        .result-content {
            background-color: #f6f8fa;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        .timestamp {
            color: #586069;
            font-style: italic;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>GitHub Stars Search Results</h1>
    <div class="timestamp">Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</div>
"""
    
    for i, result in enumerate(results, 1):
        repo = result["repository"]
        
        html_content += f"""
    <div class="result">
        <h2><a href="{repo.get('html_url', '')}" target="_blank">{repo['full_name']}</a></h2>
        <div class="result-meta">
            <span>Language: {repo.get('language', 'Unknown')}</span>
            <span>Stars: {repo.get('stargazers_count', 0)}</span>
            <span>Forks: {repo.get('forks_count', 0)}</span>
            <span class="result-score">Score: {result['score']:.2f}</span>
        </div>
"""
        
        if repo.get("description"):
            html_content += f"""
        <div class="result-description">
            {repo.get('description')}
        </div>
"""
        
        if result.get("text"):
            text_snippet = result.get('text', '')[:300] + "..." if len(result.get('text', '')) > 300 else result.get('text', '')
            html_content += f"""
        <div class="result-content">
{text_snippet}
        </div>
"""
        
        html_content += """
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

def export_to_xml(results, output_file):
    """
    Export search results to XML format.
    
    Args:
        results (list): Search results
        output_file (str): Output file path
    """
    # Create XML structure
    root = ET.Element("searchResults")
    root.set("timestamp", datetime.now().isoformat())
    
    for result in results:
        repo = result["repository"]
        
        result_elem = ET.SubElement(root, "result")
        result_elem.set("id", str(result["id"]))
        result_elem.set("score", str(result["score"]))
        
        repo_elem = ET.SubElement(result_elem, "repository")
        
        # Add repository details
        ET.SubElement(repo_elem, "id").text = str(repo["id"])
        ET.SubElement(repo_elem, "fullName").text = repo["full_name"]
        
        if repo.get("description"):
            ET.SubElement(repo_elem, "description").text = repo.get("description")
        
        if repo.get("html_url"):
            ET.SubElement(repo_elem, "url").text = repo.get("html_url")
        
        if repo.get("language"):
            ET.SubElement(repo_elem, "language").text = repo.get("language")
        
        ET.SubElement(repo_elem, "stars").text = str(repo.get("stargazers_count", 0))
        ET.SubElement(repo_elem, "forks").text = str(repo.get("forks_count", 0))
        
        if result.get("chunk_type"):
            ET.SubElement(result_elem, "chunkType").text = result.get("chunk_type")
        
        if result.get("text"):
            text_snippet = result.get('text', '')[:300] + "..." if len(result.get('text', '')) > 300 else result.get('text', '')
            ET.SubElement(result_elem, "textSnippet").text = text_snippet
    
    # Format XML with proper indentation
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_str)

def main():
    """Run the export results example."""
    parser = argparse.ArgumentParser(description="Export search results to different formats")
    parser.add_argument("--query", type=str, default="machine learning",
                        help="Search query")
    parser.add_argument("--limit", type=int, default=10,
                        help="Maximum number of results to return")
    parser.add_argument("--format", type=str, choices=["json", "csv", "markdown", "html", "xml", "all"],
                        default="all", help="Export format")
    parser.add_argument("--output-dir", type=str, default="./exports",
                        help="Output directory for exported files")
    args = parser.parse_args()
    
    console = Console()
    console.print(f"[bold]GitHub Stars Search - Export Results Example[/bold]")
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    storage_manager = StorageManager(config.get("storage", {}))
    embedding_manager = EmbeddingManager(config.get("embeddings", {}), storage_manager)
    search_engine = SearchEngine(config.get("search", {}), embedding_manager, storage_manager)
    
    # Check if we have data
    if not storage_manager.has_data():
        console.print("[bold red]No repository data found. Run 'python github_stars_search.py update' first.[/bold red]")
        return
    
    # Perform search
    console.print(f"\nSearching for: [bold]{args.query}[/bold]")
    with console.status("[bold green]Searching repositories...[/bold green]"):
        results = search_engine.search(args.query, limit=args.limit)
    
    if not results:
        console.print("[bold yellow]No results found.[/bold yellow]")
        return
    
    console.print(f"[bold green]Found {len(results)} results.[/bold green]")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export results in the requested format(s)
    if args.format == "json" or args.format == "all":
        output_file = output_dir / f"search_results_{timestamp}.json"
        export_to_json(results, output_file)
        console.print(f"[bold]Exported to JSON:[/bold] {output_file}")
    
    if args.format == "csv" or args.format == "all":
        output_file = output_dir / f"search_results_{timestamp}.csv"
        export_to_csv(results, output_file)
        console.print(f"[bold]Exported to CSV:[/bold] {output_file}")
    
    if args.format == "markdown" or args.format == "all":
        output_file = output_dir / f"search_results_{timestamp}.md"
        export_to_markdown(results, output_file)
        console.print(f"[bold]Exported to Markdown:[/bold] {output_file}")
    
    if args.format == "html" or args.format == "all":
        output_file = output_dir / f"search_results_{timestamp}.html"
        export_to_html(results, output_file)
        console.print(f"[bold]Exported to HTML:[/bold] {output_file}")
    
    if args.format == "xml" or args.format == "all":
        output_file = output_dir / f"search_results_{timestamp}.xml"
        export_to_xml(results, output_file)
        console.print(f"[bold]Exported to XML:[/bold] {output_file}")
    
    console.print("\nExample completed!")

if __name__ == "__main__":
    main()
