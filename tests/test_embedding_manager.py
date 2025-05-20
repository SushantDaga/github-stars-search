"""
Tests for the embedding manager.
"""

import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

# Import the module to test
from src.embeddings.embedding_manager import EmbeddingManager

# Sample test data
SAMPLE_CHUNKS = [
    {
        "id": "12345-0",
        "repo_id": 12345,
        "repo_name": "test-user/test-repo",
        "content": "Repository: test-user/test-repo\n\n# Test Repository",
        "chunk_type": "readme_section",
        "section_index": 0
    },
    {
        "id": "12345-1",
        "repo_id": 12345,
        "repo_name": "test-user/test-repo",
        "content": "Repository: test-user/test-repo\n\nThis is a test repository for unit tests.",
        "chunk_type": "readme_section",
        "section_index": 1
    }
]

@pytest.fixture
def mock_storage_manager():
    """Create a mock storage manager."""
    storage_manager = MagicMock()
    
    # Mock the get_embeddings_path method
    storage_manager.get_embeddings_path.return_value = Path("/tmp/embeddings")
    
    # Mock the has_embeddings method
    storage_manager.has_embeddings.return_value = False
    
    # Mock the get_repository method
    storage_manager.get_repository.return_value = {
        "id": 12345,
        "name": "test-repo",
        "full_name": "test-user/test-repo",
        "description": "A test repository"
    }
    
    return storage_manager

@pytest.fixture
def mock_embeddings():
    """Create a mock txtai Embeddings instance."""
    embeddings = MagicMock()
    
    # Mock the index method
    embeddings.index = MagicMock()
    
    # Mock the search method
    embeddings.search.return_value = [
        {
            "id": "12345-0",
            "text": "Repository: test-user/test-repo\n\n# Test Repository",
            "score": 0.9,
            "repo_id": 12345,
            "repo_name": "test-user/test-repo",
            "chunk_type": "readme_section"
        }
    ]
    
    # Mock the save and load methods
    embeddings.save = MagicMock()
    embeddings.load = MagicMock()
    
    return embeddings

@pytest.mark.unit
class TestEmbeddingManagerInit:
    """Test the initialization of the embedding manager."""
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_init_with_config(self, mock_embeddings_class, mock_storage_manager):
        """Test initialization with configuration."""
        # Set up the mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        # Create the embedding manager
        config = {
            "model": "test-model",
            "device": "cuda",
            "batch_size": 64,
            "cache_enabled": True
        }
        manager = EmbeddingManager(config, mock_storage_manager)
        
        # Assertions
        assert manager.model_name == "test-model"
        assert manager.device == "cuda"
        assert manager.batch_size == 64
        assert manager.cache_enabled is True
        assert manager.storage_manager == mock_storage_manager
        assert manager.embeddings == mock_embeddings_instance
        
        # Check that the embeddings were initialized correctly
        mock_embeddings_class.assert_called_once_with({
            "path": "test-model",
            "method": "sentence-transformers",
            "device": "cuda",
            "content": True
        })
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_init_with_empty_config(self, mock_embeddings_class, mock_storage_manager):
        """Test initialization with empty configuration."""
        # Set up the mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        # Create the embedding manager
        manager = EmbeddingManager({}, mock_storage_manager)
        
        # Assertions
        assert manager.model_name == "BAAI/bge-small-en-v1.5"  # Default value
        assert manager.device == "cpu"  # Default value
        assert manager.batch_size == 32  # Default value
        assert manager.cache_enabled is True  # Default value
        assert manager.storage_manager == mock_storage_manager
        assert manager.embeddings == mock_embeddings_instance
        
        # Check that the embeddings were initialized correctly
        mock_embeddings_class.assert_called_once_with({
            "path": "BAAI/bge-small-en-v1.5",
            "method": "sentence-transformers",
            "device": "cpu",
            "content": True
        })
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_init_loads_existing_index(self, mock_embeddings_class, mock_storage_manager):
        """Test that initialization loads an existing index if available."""
        # Set up the mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        # Mock the Path.exists method to return True
        with patch.object(Path, "exists", return_value=True):
            # Create the embedding manager
            manager = EmbeddingManager({}, mock_storage_manager)
            
            # Check that the load method was called
            mock_embeddings_instance.load.assert_called_once()

@pytest.mark.unit
class TestGenerateEmbeddings:
    """Test the generate_embeddings method."""
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_generate_embeddings(self, mock_embeddings_class, mock_storage_manager):
        """Test generating embeddings for repository chunks."""
        # Set up the mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        # Create the embedding manager
        manager = EmbeddingManager({}, mock_storage_manager)
        
        # Call the method
        result = manager.generate_embeddings(12345, SAMPLE_CHUNKS)
        
        # Assertions
        assert result is True
        
        # Check that the index method was called with the correct arguments
        mock_embeddings_instance.index.assert_called_once()
        args, kwargs = mock_embeddings_instance.index.call_args
        
        # Check the documents passed to the index method
        documents = args[0]
        assert len(documents) == 2
        assert documents[0]["id"] == SAMPLE_CHUNKS[0]["id"]
        assert documents[0]["text"] == SAMPLE_CHUNKS[0]["content"]
        assert documents[0]["repo_id"] == SAMPLE_CHUNKS[0]["repo_id"]
        assert documents[0]["repo_name"] == SAMPLE_CHUNKS[0]["repo_name"]
        assert documents[0]["chunk_type"] == SAMPLE_CHUNKS[0]["chunk_type"]
        assert documents[0]["section_index"] == SAMPLE_CHUNKS[0]["section_index"]
        
        # Check that the save method was called
        mock_embeddings_instance.save.assert_called_once()
        
        # Check that the repository was marked as embedded
        mock_storage_manager.mark_repository_embedded.assert_called_once_with(12345)
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_generate_embeddings_no_chunks(self, mock_embeddings_class, mock_storage_manager):
        """Test generating embeddings with no chunks."""
        # Set up the mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        # Create the embedding manager
        manager = EmbeddingManager({}, mock_storage_manager)
        
        # Call the method with empty chunks
        result = manager.generate_embeddings(12345, [])
        
        # Assertions
        assert result is False
        
        # Check that the index method was not called
        mock_embeddings_instance.index.assert_not_called()
        
        # Check that the save method was not called
        mock_embeddings_instance.save.assert_not_called()
        
        # Check that the repository was not marked as embedded
        mock_storage_manager.mark_repository_embedded.assert_not_called()
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_generate_embeddings_already_embedded(self, mock_embeddings_class, mock_storage_manager):
        """Test generating embeddings for a repository that already has embeddings."""
        # Set up the mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        # Mock the has_embeddings method to return True
        mock_storage_manager.has_embeddings.return_value = True
        
        # Create the embedding manager
        manager = EmbeddingManager({}, mock_storage_manager)
        
        # Call the method
        result = manager.generate_embeddings(12345, SAMPLE_CHUNKS)
        
        # Assertions
        assert result is True
        
        # Check that the index method was not called
        mock_embeddings_instance.index.assert_not_called()
        
        # Check that the save method was not called
        mock_embeddings_instance.save.assert_not_called()
        
        # Check that the repository was not marked as embedded
        mock_storage_manager.mark_repository_embedded.assert_not_called()
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_generate_embeddings_cache_disabled(self, mock_embeddings_class, mock_storage_manager):
        """Test generating embeddings with cache disabled."""
        # Set up the mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        # Mock the has_embeddings method to return True
        mock_storage_manager.has_embeddings.return_value = True
        
        # Create the embedding manager with cache disabled
        config = {"cache_enabled": False}
        manager = EmbeddingManager(config, mock_storage_manager)
        
        # Call the method
        result = manager.generate_embeddings(12345, SAMPLE_CHUNKS)
        
        # Assertions
        assert result is True
        
        # Check that the index method was called even though the repository has embeddings
        mock_embeddings_instance.index.assert_called_once()
        
        # Check that the save method was called
        mock_embeddings_instance.save.assert_called_once()
        
        # Check that the repository was marked as embedded
        mock_storage_manager.mark_repository_embedded.assert_called_once_with(12345)

@pytest.mark.unit
class TestSearch:
    """Test the search method."""
    
    def test_search(self, mock_storage_manager, mock_embeddings):
        """Test searching for similar content."""
        # Create the embedding manager with the mock embeddings
        manager = EmbeddingManager({}, mock_storage_manager)
        manager.embeddings = mock_embeddings
        
        # Call the method
        results = manager.search("test query")
        
        # Assertions
        assert len(results) == 1
        assert results[0]["id"] == "12345-0"
        assert results[0]["score"] == 0.9
        assert "repository" in results[0]
        assert results[0]["repository"]["id"] == 12345
        assert results[0]["repository"]["full_name"] == "test-user/test-repo"
        
        # Check that the search method was called with the correct arguments
        mock_embeddings.search.assert_called_once_with("test query", 10)
    
    def test_search_with_limit(self, mock_storage_manager, mock_embeddings):
        """Test searching with a custom limit."""
        # Create the embedding manager with the mock embeddings
        manager = EmbeddingManager({}, mock_storage_manager)
        manager.embeddings = mock_embeddings
        
        # Call the method with a custom limit
        results = manager.search("test query", limit=5)
        
        # Assertions
        assert len(results) == 1
        
        # Check that the search method was called with the correct arguments
        mock_embeddings.search.assert_called_once_with("test query", 5)
    
    def test_search_no_embeddings(self, mock_storage_manager):
        """Test searching when embeddings are not initialized."""
        # Create the embedding manager with no embeddings
        manager = EmbeddingManager({}, mock_storage_manager)
        manager.embeddings = None
        
        # Call the method
        results = manager.search("test query")
        
        # Assertions
        assert results == []
    
    def test_search_no_results(self, mock_storage_manager, mock_embeddings):
        """Test searching with no results."""
        # Set up the mock to return no results
        mock_embeddings.search.return_value = []
        
        # Create the embedding manager with the mock embeddings
        manager = EmbeddingManager({}, mock_storage_manager)
        manager.embeddings = mock_embeddings
        
        # Call the method
        results = manager.search("test query")
        
        # Assertions
        assert results == []
    
    def test_search_repository_not_found(self, mock_storage_manager, mock_embeddings):
        """Test searching when a repository is not found."""
        # Set up the mock to return a repository that doesn't exist
        mock_embeddings.search.return_value = [
            {
                "id": "99999-0",
                "text": "Repository: not-found/repo\n\n# Test Repository",
                "score": 0.9,
                "repo_id": 99999,
                "repo_name": "not-found/repo",
                "chunk_type": "readme_section"
            }
        ]
        
        # Mock the get_repository method to return None
        mock_storage_manager.get_repository.return_value = None
        
        # Create the embedding manager with the mock embeddings
        manager = EmbeddingManager({}, mock_storage_manager)
        manager.embeddings = mock_embeddings
        
        # Call the method
        results = manager.search("test query")
        
        # Assertions
        assert results == []

@pytest.mark.unit
class TestGetModelInfo:
    """Test the get_model_info method."""
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_get_model_info(self, mock_embeddings_class, mock_storage_manager):
        """Test getting model information."""
        # Set up the mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        # Create the embedding manager
        config = {
            "model": "test-model",
            "device": "cuda",
            "batch_size": 64
        }
        manager = EmbeddingManager(config, mock_storage_manager)
        
        # Call the method
        info = manager.get_model_info()
        
        # Assertions
        assert info["model_name"] == "test-model"
        assert info["device"] == "cuda"
        assert info["batch_size"] == 64
