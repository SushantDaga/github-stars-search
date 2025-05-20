"""
Embedding manager for GitHub Stars Search.
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sentence_transformers # needed to get rid of OMP error when dealing with txtai
from txtai.embeddings import Embeddings

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manager for generating and storing embeddings.
    """
    
    def __init__(self, config: Dict[str, Any], storage_manager):
        """
        Initialize the embedding manager.
        
        Args:
            config (dict): Configuration dictionary
            storage_manager: Storage manager instance
        """
        self.model_name = config.get("model", "BAAI/bge-small-en-v1.5")
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 32)
        self.cache_enabled = config.get("cache_enabled", True)
        
        self.storage_manager = storage_manager
        self.embeddings = None
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """
        Initialize the embeddings model.
        """
        logger.info(f"Initializing embeddings with model: {self.model_name}")
        
        try:
            # Create embeddings instance
            self.embeddings = Embeddings({
                "path": self.model_name,
                "method": "sentence-transformers",
                "device": self.device,
                "content": True
            })
            
            # Load existing index if available
            index_path = self.storage_manager.get_embeddings_path() / "index"
            embeddings_file = index_path / "embeddings"
            
            if index_path.exists() and embeddings_file.exists():
                logger.info("Loading existing embeddings index")
                try:
                    self.embeddings.load(str(index_path))
                except Exception as e:
                    logger.warning(f"Failed to load existing index: {str(e)}. Creating a new one.")
                    # Create an empty index
                    self.embeddings.index([])
                    self._save_embeddings()
            elif index_path.exists():
                # Index directory exists but embeddings file doesn't
                logger.warning("Index directory exists but embeddings file is missing. Creating a new index.")
                # Create an empty index
                self.embeddings.index([])
                self._save_embeddings()
            
            logger.info("Embeddings initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def generate_embeddings(self, repo_id: int, chunks: List[Dict[str, Any]]) -> bool:
        """
        Generate embeddings for repository chunks.
        
        Args:
            repo_id (int): Repository ID
            chunks (list): List of content chunks
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Generating embeddings for repository {repo_id}")
        
        try:
            # Skip if no chunks
            if not chunks:
                logger.warning(f"No chunks to embed for repository {repo_id}")
                return False
            
            # Check if embeddings already exist and are up to date
            if self.cache_enabled and self.storage_manager.has_embeddings(repo_id):
                logger.info(f"Embeddings already exist for repository {repo_id}")
                return True
            
            # Prepare documents for embedding
            documents = []
            for chunk in chunks:
                # Create document with metadata
                document = {
                    "id": chunk["id"],
                    "text": chunk["content"],
                    "repo_id": chunk["repo_id"],
                    "repo_name": chunk["repo_name"],
                    "chunk_type": chunk["chunk_type"]
                }
                
                # Add additional metadata
                if "section_index" in chunk:
                    document["section_index"] = chunk["section_index"]
                if "window_index" in chunk:
                    document["window_index"] = chunk["window_index"]
                
                documents.append(document)
            
            # Generate embeddings
            self.embeddings.index(documents, self.batch_size)
            
            # Save embeddings
            self._save_embeddings(repo_id)
            
            return True
        
        except Exception as e:
            logger.error(f"Error generating embeddings for repository {repo_id}: {str(e)}")
            return False
    
    def _save_embeddings(self, repo_id: Optional[int] = None):
        """
        Save embeddings to disk.
        
        Args:
            repo_id (int, optional): Repository ID to save embeddings for
        """
        try:
            # Create embeddings directory if it doesn't exist
            embeddings_dir = self.storage_manager.get_embeddings_path()
            embeddings_dir.mkdir(exist_ok=True, parents=True)
            
            # Create index directory if it doesn't exist
            index_path = embeddings_dir / "index"
            index_path.mkdir(exist_ok=True, parents=True)
            
            # Save embeddings index
            try:
                self.embeddings.save(str(index_path))
                logger.info(f"Embeddings saved to {index_path}")
            except AttributeError as e:
                if "'NoneType' object has no attribute 'commit'" in str(e):
                    logger.warning("Database connection issue detected. Attempting to fix...")
                    
                    # Force initialize the database
                    try:
                        # Create a simple document to initialize the database
                        self.embeddings.index([{
                            "id": "init-doc",
                            "text": "Initialization document",
                            "repo_id": 0,
                            "repo_name": "initialization",
                            "chunk_type": "init"
                        }])
                        
                        # Try saving again
                        self.embeddings.save(str(index_path))
                        logger.info("Successfully initialized and saved the index")
                    except Exception as init_error:
                        logger.error(f"Failed to initialize database: {str(init_error)}")
                else:
                    raise
            
            # If repo_id is provided, mark as embedded
            if repo_id is not None:
                self.storage_manager.mark_repository_embedded(repo_id)
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar content using embeddings.
        
        Args:
            query (str): Search query
            limit (int, optional): Maximum number of results to return
        
        Returns:
            list: List of search results with scores
        """
        logger.info(f"Searching embeddings for: {query}")
        
        try:
            # Check if embeddings are initialized
            if self.embeddings is None:
                logger.error("Embeddings not initialized")
                return []
            
            # Check if index exists
            index_path = self.storage_manager.get_embeddings_path() / "index"
            if not index_path.exists():
                logger.warning("No embeddings index found. Creating empty index.")
                # Create an empty index
                self.embeddings.index([])
                self._save_embeddings()
                return []
            
            try:
                # Execute search
                results = self.embeddings.search(query, limit)
            except Exception as search_error:
                error_msg = str(search_error)
                # Handle specific database errors
                if "No indexes available" in error_msg:
                    logger.warning("No indexes available for search. Attempting to fix...")
                    
                    # Force initialize the database
                    try:
                        # Create a simple document to initialize the database
                        self.embeddings.index([{
                            "id": "init-doc",
                            "text": "Initialization document",
                            "repo_id": 0,
                            "repo_name": "initialization",
                            "chunk_type": "init"
                        }])
                        
                        # Try saving
                        self._save_embeddings()
                        
                        # Try searching again
                        try:
                            results = self.embeddings.search(query, limit)
                            logger.info("Successfully fixed the index and performed search")
                            # Continue with normal processing
                        except Exception as retry_error:
                            logger.error(f"Failed to search after fixing index: {str(retry_error)}")
                            return []
                    except Exception as init_error:
                        logger.error(f"Failed to initialize database: {str(init_error)}")
                        return []
                elif "no such table: sections" in error_msg or "'NoneType' object has no attribute" in error_msg:
                    logger.warning(f"Database schema issue detected: {error_msg}. Recreating index.")
                    
                    # Remove the corrupted index file and any related files
                    embeddings_dir = self.storage_manager.get_embeddings_path()
                    index_path = embeddings_dir / "index"
                    
                    # Clean up all files in the embeddings directory
                    try:
                        for file in embeddings_dir.glob("*"):
                            if file.is_file():
                                # Create a backup directory if it doesn't exist
                                backup_dir = embeddings_dir / "backup"
                                backup_dir.mkdir(exist_ok=True)
                                
                                # Move the file to the backup directory
                                backup_file = backup_dir / f"{file.name}.{int(time.time())}"
                                file.rename(backup_file)
                                logger.info(f"Moved {file} to {backup_file}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up embeddings directory: {str(e)}")
                    
                    # Re-initialize the embeddings object
                    self.embeddings = Embeddings({
                        "path": self.model_name,
                        "method": "sentence-transformers",
                        "device": self.device,
                        "content": True
                    })
                    
                    # Create a new empty index
                    self.embeddings.index([])
                    self._save_embeddings()
                    return []
                else:
                    # Re-raise other errors
                    raise
            
            # Format results
            formatted_results = []
            for result in results:
                # Get repository data
                # Handle both string and integer repo_id
                if isinstance(result, dict) and "repo_id" in result:
                    repo_id = result["repo_id"]
                    repo_data = self.storage_manager.get_repository(int(repo_id))
                    
                    if repo_data:
                        formatted_results.append({
                            "id": result["id"],
                            "score": result["score"],
                            "text": result["text"],
                            "repository": repo_data,
                            "chunk_type": result["chunk_type"]
                        })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching embeddings: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model.
        
        Returns:
            dict: Model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size
        }
