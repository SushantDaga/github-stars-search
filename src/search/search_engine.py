"""
Search engine for GitHub Stars Search.
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import sentence_transformers # needed to get rid of OMP error when dealing with txtai
from txtai.pipeline import Similarity
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Search engine for GitHub repositories.
    """
    
    def __init__(self, config: Dict[str, Any], embedding_manager, storage_manager):
        """
        Initialize the search engine.
        
        Args:
            config (dict): Configuration dictionary
            embedding_manager: Embedding manager instance
            storage_manager: Storage manager instance
        """
        self.hybrid_enabled = config.get("hybrid_enabled", True)
        self.neural_weight = config.get("neural_weight", 0.7)
        self.keyword_weight = config.get("keyword_weight", 0.3)
        self.max_results = config.get("max_results", 20)
        self.min_score = config.get("min_score", 0.2)
        
        self.embedding_manager = embedding_manager
        self.storage_manager = storage_manager
        
        # Initialize BM25 index
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_repo_map = {}
        
        # Initialize BM25 index if hybrid search is enabled
        if self.hybrid_enabled:
            self._initialize_bm25_index()
    
    def _initialize_bm25_index(self):
        """
        Initialize the BM25 index for keyword search.
        """
        logger.info("Initializing BM25 index")
        
        try:
            # Get all repository IDs
            repo_ids = self.storage_manager.get_all_repositories()
            
            if not repo_ids:
                logger.warning("No repositories found for BM25 indexing")
                return
            
            # Prepare documents for BM25
            documents = []
            repo_map = {}
            
            # Check if repo_ids is a dictionary or a list
            if isinstance(repo_ids, dict):
                # It's a dictionary of repo_id -> repo_data
                repositories = repo_ids
                for repo_id, repo_data in repositories.items():
                    # Get README content
                    readme = self.storage_manager.get_repository_readme(repo_id)
                    
                    if readme:
                        # Clean and tokenize text
                        text = self._preprocess_text(readme)
                        tokens = text.split()
                        
                        # Add to documents
                        doc_id = len(documents)
                        documents.append(tokens)
                        repo_map[doc_id] = repo_id
                    
                    # Add description if available
                    description = repo_data.get("description")
                    if description:
                        # Clean and tokenize text
                        text = self._preprocess_text(description)
                        tokens = text.split()
                        
                        # Add to documents
                        doc_id = len(documents)
                        documents.append(tokens)
                        repo_map[doc_id] = repo_id
            else:
                # It's a list of repo_ids
                for repo_id in repo_ids:
                    # Get repository data
                    repo_data = self.storage_manager.get_repository(repo_id)
                    if not repo_data:
                        logger.warning(f"Repository data not found for ID {repo_id}")
                        continue
                    
                    # Get README content
                    readme = self.storage_manager.get_repository_readme(repo_id)
                    
                    if readme:
                        # Clean and tokenize text
                        text = self._preprocess_text(readme)
                        tokens = text.split()
                        
                        # Add to documents
                        doc_id = len(documents)
                        documents.append(tokens)
                        repo_map[doc_id] = repo_id
                    
                    # Add description if available
                    description = repo_data.get("description")
                    if description:
                        # Clean and tokenize text
                        text = self._preprocess_text(description)
                        tokens = text.split()
                        
                        # Add to documents
                        doc_id = len(documents)
                        documents.append(tokens)
                        repo_map[doc_id] = repo_id
            
            # Create BM25 index
            if documents:
                self.bm25_index = BM25Okapi(documents)
                self.bm25_documents = documents
                self.bm25_repo_map = repo_map
                
                logger.info(f"BM25 index initialized with {len(documents)} documents")
            else:
                logger.warning("No documents found for BM25 indexing")
        
        except Exception as e:
            logger.error(f"Error initializing BM25 index: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for BM25 indexing.
        
        Args:
            text (str): Text to preprocess
        
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for repositories matching the query.
        
        Args:
            query (str): Search query
            filters (dict, optional): Filters to apply to results
            limit (int, optional): Maximum number of results to return
        
        Returns:
            list: List of search results
        """
        logger.info(f"Searching for: {query}")
        
        # Set limit
        if limit is None:
            limit = self.max_results
        
        # Initialize results
        neural_results = []
        keyword_results = []
        
        # Perform neural search
        neural_results = self.embedding_manager.search(query, limit=limit * 2)
        
        # Perform keyword search if hybrid enabled
        if self.hybrid_enabled and self.bm25_index is not None:
            keyword_results = self._keyword_search(query, limit=limit * 2)
        
        # Merge results
        if self.hybrid_enabled and keyword_results:
            merged_results = self._merge_results(neural_results, keyword_results)
        else:
            merged_results = neural_results
        
        # Apply filters
        if filters:
            merged_results = self._apply_filters(merged_results, filters)
        
        # Limit results
        merged_results = merged_results[:limit]
        
        return merged_results
    
    def _keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search using BM25.
        
        Args:
            query (str): Search query
            limit (int, optional): Maximum number of results to return
        
        Returns:
            list: List of search results
        """
        logger.info(f"Performing keyword search for: {query}")
        
        try:
            # Preprocess query
            processed_query = self._preprocess_text(query)
            query_tokens = processed_query.split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top results
            top_indices = np.argsort(scores)[::-1][:limit]
            
            # Format results
            results = []
            for idx in top_indices:
                score = scores[idx]
                
                # Skip low scores
                if score < self.min_score:
                    continue
                
                # Get repository data
                repo_id = self.bm25_repo_map.get(idx)
                if repo_id:
                    repo_data = self.storage_manager.get_repository(repo_id)
                    
                    if repo_data:
                        results.append({
                            "id": f"bm25-{repo_id}-{idx}",
                            "score": float(score),
                            "text": " ".join(self.bm25_documents[idx][:50]) + "...",
                            "repository": repo_data,
                            "chunk_type": "bm25"
                        })
            
            return results
        
        except Exception as e:
            logger.error(f"Error performing keyword search: {str(e)}")
            return []
    
    def _merge_results(self, neural_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge neural and keyword search results.
        
        Args:
            neural_results (list): Neural search results
            keyword_results (list): Keyword search results
        
        Returns:
            list: Merged search results
        """
        logger.info("Merging neural and keyword search results")
        
        # Create a map of repository IDs to results
        repo_results = {}
        
        # Process neural results
        for result in neural_results:
            repo_id = result["repository"]["id"]
            
            if repo_id not in repo_results:
                repo_results[repo_id] = {
                    "repository": result["repository"],
                    "neural_score": result["score"],
                    "keyword_score": 0.0,
                    "text": result["text"],
                    "chunk_type": result["chunk_type"]
                }
            else:
                # Update if better score
                if result["score"] > repo_results[repo_id]["neural_score"]:
                    repo_results[repo_id]["neural_score"] = result["score"]
                    repo_results[repo_id]["text"] = result["text"]
                    repo_results[repo_id]["chunk_type"] = result["chunk_type"]
        
        # Process keyword results
        for result in keyword_results:
            repo_id = result["repository"]["id"]
            
            if repo_id not in repo_results:
                repo_results[repo_id] = {
                    "repository": result["repository"],
                    "neural_score": 0.0,
                    "keyword_score": result["score"],
                    "text": result["text"],
                    "chunk_type": result["chunk_type"]
                }
            else:
                # Update if better score
                if result["score"] > repo_results[repo_id]["keyword_score"]:
                    repo_results[repo_id]["keyword_score"] = result["score"]
                    
                    # Only update text if neural score is low
                    if repo_results[repo_id]["neural_score"] < 0.5:
                        repo_results[repo_id]["text"] = result["text"]
                        repo_results[repo_id]["chunk_type"] = result["chunk_type"]
        
        # Calculate combined scores
        merged_results = []
        for repo_id, result in repo_results.items():
            # Normalize keyword score (BM25 scores can be > 1)
            keyword_score = min(result["keyword_score"], 10.0) / 10.0
            
            # Calculate combined score
            combined_score = (
                self.neural_weight * result["neural_score"] +
                self.keyword_weight * keyword_score
            )
            
            # Skip low scores
            if combined_score < self.min_score:
                continue
            
            # Add to results
            merged_results.append({
                "id": f"merged-{repo_id}",
                "score": combined_score,
                "text": result["text"],
                "repository": result["repository"],
                "chunk_type": result["chunk_type"],
                "neural_score": result["neural_score"],
                "keyword_score": keyword_score
            })
        
        # Sort by combined score
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        return merged_results
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply filters to search results.
        
        Args:
            results (list): Search results
            filters (dict): Filters to apply
        
        Returns:
            list: Filtered search results
        """
        filtered_results = []
        
        for result in results:
            repo = result["repository"]
            include = True
            
            # Apply filters
            for field, filter_value in filters.items():
                if field in repo:
                    field_value = repo[field]
                    
                    # Handle different filter types
                    if isinstance(filter_value, dict):
                        # Range filter
                        if "min" in filter_value and field_value < filter_value["min"]:
                            include = False
                            break
                        if "max" in filter_value and field_value > filter_value["max"]:
                            include = False
                            break
                    else:
                        # Exact match filter
                        if field_value != filter_value:
                            include = False
                            break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
