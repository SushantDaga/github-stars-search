"""
Storage manager for GitHub Stars Search.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Manager for storing and retrieving repository data and embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the storage manager.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.compress_data = config.get("compress_data", True)
        self.backup_enabled = config.get("backup_enabled", True)
        self.max_backups = config.get("max_backups", 3)
        
        # Set up paths
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / "data"
        self.repositories_path = self.data_path / "repositories"
        self.embeddings_path = self.data_path / "embeddings"
        self.index_path = self.data_path / "index"
        
        # Create directories if they don't exist
        self.repositories_path.mkdir(exist_ok=True, parents=True)
        self.embeddings_path.mkdir(exist_ok=True, parents=True)
        self.index_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize repository index
        self.repository_index = self._load_repository_index()
    
    def _load_repository_index(self) -> Dict[int, Dict[str, Any]]:
        """
        Load the repository index from disk.
        
        Returns:
            dict: Repository index
        """
        index_file = self.index_path / "repositories.json"
        
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading repository index: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_repository_index(self):
        """
        Save the repository index to disk.
        """
        index_file = self.index_path / "repositories.json"
        
        try:
            with open(index_file, "w") as f:
                json.dump(self.repository_index, f)
        except Exception as e:
            logger.error(f"Error saving repository index: {str(e)}")
    
    def store_repository(self, repo: Dict[str, Any], readme: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        Store repository data.
        
        Args:
            repo (dict): Repository metadata
            readme (str): README content
            chunks (list): Content chunks
        
        Returns:
            bool: True if successful, False otherwise
        """
        repo_id = repo["id"]
        logger.info(f"Storing repository {repo_id}")
        
        try:
            # Create repository directory
            repo_dir = self.repositories_path / str(repo_id)
            repo_dir.mkdir(exist_ok=True)
            
            # Backup existing data if enabled
            if self.backup_enabled and repo_dir.exists():
                self._backup_repository(repo_id)
            
            # Store metadata
            with open(repo_dir / "metadata.json", "w") as f:
                json.dump(repo, f)
            
            # Store README
            with open(repo_dir / "readme.md", "w") as f:
                f.write(readme)
            
            # Store chunks
            with open(repo_dir / "chunks.json", "w") as f:
                json.dump(chunks, f)
            
            # Update index
            self.repository_index[str(repo_id)] = {
                "id": repo_id,
                "full_name": repo["full_name"],
                "updated_at": repo["updated_at"],
                "embedded": False,
                "stored_at": datetime.now().isoformat()
            }
            
            # Save index
            self._save_repository_index()
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing repository {repo_id}: {str(e)}")
            return False
    
    def _backup_repository(self, repo_id: int):
        """
        Backup repository data.
        
        Args:
            repo_id (int): Repository ID
        """
        repo_dir = self.repositories_path / str(repo_id)
        backup_dir = self.repositories_path / f"{repo_id}_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        try:
            # Copy repository directory
            shutil.copytree(repo_dir, backup_dir)
            
            # Remove old backups if needed
            self._cleanup_backups(repo_id)
            
            logger.info(f"Backed up repository {repo_id}")
        
        except Exception as e:
            logger.error(f"Error backing up repository {repo_id}: {str(e)}")
    
    def _cleanup_backups(self, repo_id: int):
        """
        Clean up old repository backups.
        
        Args:
            repo_id (int): Repository ID
        """
        # Get all backup directories
        backup_dirs = list(self.repositories_path.glob(f"{repo_id}_backup_*"))
        
        # Sort by creation time (oldest first)
        backup_dirs.sort(key=lambda x: x.stat().st_ctime)
        
        # Remove oldest backups if there are too many
        while len(backup_dirs) > self.max_backups:
            oldest = backup_dirs.pop(0)
            shutil.rmtree(oldest)
            logger.info(f"Removed old backup: {oldest}")
    
    def get_repository(self, repo_id: int) -> Optional[Dict[str, Any]]:
        """
        Get repository metadata.
        
        Args:
            repo_id (int): Repository ID
        
        Returns:
            dict: Repository metadata or None if not found
        """
        repo_dir = self.repositories_path / str(repo_id)
        metadata_file = repo_dir / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading repository metadata for {repo_id}: {str(e)}")
                return None
        else:
            return None
    
    def get_repository_readme(self, repo_id: int) -> Optional[str]:
        """
        Get repository README content.
        
        Args:
            repo_id (int): Repository ID
        
        Returns:
            str: README content or None if not found
        """
        repo_dir = self.repositories_path / str(repo_id)
        readme_file = repo_dir / "readme.md"
        
        if readme_file.exists():
            try:
                with open(readme_file, "r") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error loading repository README for {repo_id}: {str(e)}")
                return None
        else:
            return None
    
    def get_repository_chunks(self, repo_id: int) -> Optional[List[Dict[str, Any]]]:
        """
        Get repository content chunks.
        
        Args:
            repo_id (int): Repository ID
        
        Returns:
            list: Content chunks or None if not found
        """
        repo_dir = self.repositories_path / str(repo_id)
        chunks_file = repo_dir / "chunks.json"
        
        if chunks_file.exists():
            try:
                with open(chunks_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading repository chunks for {repo_id}: {str(e)}")
                return None
        else:
            return None
    
    def has_repository(self, repo_id: int) -> bool:
        """
        Check if repository exists.
        
        Args:
            repo_id (int): Repository ID
        
        Returns:
            bool: True if repository exists, False otherwise
        """
        return str(repo_id) in self.repository_index
    
    def is_repository_outdated(self, repo: Dict[str, Any]) -> bool:
        """
        Check if repository is outdated.
        
        Args:
            repo (dict): Repository metadata
        
        Returns:
            bool: True if repository is outdated, False otherwise
        """
        repo_id = repo["id"]
        
        if str(repo_id) not in self.repository_index:
            return True
        
        # Check if updated_at is different
        stored_updated_at = self.repository_index[str(repo_id)]["updated_at"]
        current_updated_at = repo["updated_at"]
        
        return stored_updated_at != current_updated_at
    
    def mark_repository_embedded(self, repo_id: int):
        """
        Mark repository as embedded.
        
        Args:
            repo_id (int): Repository ID
        """
        if str(repo_id) in self.repository_index:
            self.repository_index[str(repo_id)]["embedded"] = True
            self._save_repository_index()
    
    def has_embeddings(self, repo_id: int) -> bool:
        """
        Check if repository has embeddings.
        
        Args:
            repo_id (int): Repository ID
        
        Returns:
            bool: True if repository has embeddings, False otherwise
        """
        if str(repo_id) not in self.repository_index:
            return False
        
        return self.repository_index[str(repo_id)].get("embedded", False)
    
    def get_embeddings_path(self) -> Path:
        """
        Get the path to the embeddings directory.
        
        Returns:
            Path: Embeddings directory path
        """
        return self.embeddings_path
    
    def get_all_repositories(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all repositories.
        
        Returns:
            dict: Dictionary of repository ID to metadata
        """
        repositories = {}
        
        for repo_id in self.repository_index:
            repo_data = self.get_repository(int(repo_id))
            if repo_data:
                repositories[int(repo_id)] = repo_data
        
        return repositories
    
    def get_repository_count(self) -> int:
        """
        Get the number of repositories.
        
        Returns:
            int: Number of repositories
        """
        return len(self.repository_index)
    
    def get_embedding_count(self) -> int:
        """
        Get the number of embedded repositories.
        
        Returns:
            int: Number of embedded repositories
        """
        return sum(1 for repo_id in self.repository_index if self.repository_index[repo_id].get("embedded", False))
    
    def has_data(self) -> bool:
        """
        Check if there is any data.
        
        Returns:
            bool: True if there is data, False otherwise
        """
        return len(self.repository_index) > 0
