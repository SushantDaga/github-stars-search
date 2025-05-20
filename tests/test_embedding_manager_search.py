"""
Tests for the embedding manager search functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import shutil
from pathlib import Path

# Import the module to test
from src.embeddings.embedding_manager import EmbeddingManager
from src.storage.storage_manager import StorageManager

@pytest.fixture
def mock_storage_manager():
    """Create a mock storage manager."""
    mock = MagicMock(spec=StorageManager)
    
    # Set up the get_embeddings_path method
    embeddings_path = Path("tests/test_data/embeddings")
    mock.get_embeddings_path.return_value = embeddings_path
    
    # Set up the get_repository method
    mock.get_repository.return_value = {
        "id": 12345,
        "name": "test-repo",
        "full_name": "test-user/test-repo",
        "description": "A test repository"
    }
    
    return mock

@pytest.fixture
def clean_test_data():
    """Clean up test data before and after tests."""
    # Create test data directory
    test_data_dir = Path("tests/test_data/embeddings")
    test_data_dir.mkdir(exist_ok=True, parents=True)
    
    yield
    
    # Clean up after tests
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)

@pytest.mark.unit
class TestEmbeddingManagerSearch:
    """Test the search functionality of the embedding manager."""
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_search_no_index(self, mock_embeddings_class, mock_storage_manager, clean_test_data):
        """Test searching when no index exists."""
        # Set up the mock
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        # Set up the storage manager to return a path that doesn't exist
        mock_storage_manager.get_embeddings_path.return_value = Path("tests/test_data/embeddings_nonexistent")
        
        # Create the embedding manager
        config = {
            "model": "test-model",
            "device": "cpu",
            "batch_size": 32
        }
        manager = EmbeddingManager(config, mock_storage_manager)
        
        # Call the method
        results = manager.search("test query")
        
        # Assertions
        assert results == []
        mock_embeddings.index.assert_called_once()
        mock_embeddings.save.assert_called_once()
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_search_with_results(self, mock_embeddings_class, mock_storage_manager, clean_test_data):
        """Test searching with results."""
        # Set up the mock
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        # Mock search results
        mock_embeddings.search.return_value = [
            {
                "id": "12345-0",
                "score": 0.9,
                "text": "Test content",
                "repo_id": 12345,
                "chunk_type": "readme_section"
            }
        ]
        
        # Create the embedding manager
        config = {
            "model": "test-model",
            "device": "cpu",
            "batch_size": 32
        }
        manager = EmbeddingManager(config, mock_storage_manager)
        
        # Create a fake index file
        index_path = mock_storage_manager.get_embeddings_path() / "index"
        index_path.parent.mkdir(exist_ok=True, parents=True)
        index_path.touch()
        
        # Call the method
        results = manager.search("test query")
        
        # Assertions
        assert len(results) == 1
        assert results[0]["score"] == 0.9
        assert results[0]["repository"]["id"] == 12345
        assert results[0]["chunk_type"] == "readme_section"
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_search_with_string_repo_id(self, mock_embeddings_class, mock_storage_manager, clean_test_data):
        """Test searching with string repo_id in results."""
        # Set up the mock
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        # Mock search results with string repo_id
        mock_embeddings.search.return_value = [
            {
                "id": "12345-0",
                "score": 0.9,
                "text": "Test content",
                "repo_id": "12345",  # String instead of int
                "chunk_type": "readme_section"
            }
        ]
        
        # Create the embedding manager
        config = {
            "model": "test-model",
            "device": "cpu",
            "batch_size": 32
        }
        manager = EmbeddingManager(config, mock_storage_manager)
        
        # Create a fake index file
        index_path = mock_storage_manager.get_embeddings_path() / "index"
        index_path.parent.mkdir(exist_ok=True, parents=True)
        index_path.touch()
        
        # Call the method
        results = manager.search("test query")
        
        # Assertions
        assert len(results) == 1
        assert results[0]["score"] == 0.9
        assert results[0]["repository"]["id"] == 12345
        assert results[0]["chunk_type"] == "readme_section"
        
        # Verify that get_repository was called with int
        mock_storage_manager.get_repository.assert_called_with(12345)
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_search_with_error(self, mock_embeddings_class, mock_storage_manager, clean_test_data):
        """Test searching with an error."""
        # Set up the mock to raise an exception
        mock_embeddings = MagicMock()
        mock_embeddings.search.side_effect = Exception("Test error")
        mock_embeddings_class.return_value = mock_embeddings
        
        # Create the embedding manager
        config = {
            "model": "test-model",
            "device": "cpu",
            "batch_size": 32
        }
        manager = EmbeddingManager(config, mock_storage_manager)
        
        # Create a fake index file
        index_path = mock_storage_manager.get_embeddings_path() / "index"
        index_path.parent.mkdir(exist_ok=True, parents=True)
        index_path.touch()
        
        # Call the method
        results = manager.search("test query")
        
        # Assertions
        assert results == []
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    @patch("src.embeddings.embedding_manager.time")
    def test_search_with_sections_table_error(self, mock_time, mock_embeddings_class, mock_storage_manager, clean_test_data):
        """Test searching with a 'no such table: sections' error."""
        # Set up the mocks
        mock_time.time.return_value = 12345
        
        mock_embeddings = MagicMock()
        mock_embeddings.search.side_effect = Exception("no such table: sections")
        mock_embeddings_class.return_value = mock_embeddings
        
        # Create the embedding manager
        config = {
            "model": "test-model",
            "device": "cpu",
            "batch_size": 32
        }
        manager = EmbeddingManager(config, mock_storage_manager)
        
        # Create a fake index file
        embeddings_path = mock_storage_manager.get_embeddings_path()
        embeddings_path.mkdir(exist_ok=True, parents=True)
        index_path = embeddings_path / "index"
        index_path.touch()
        
        # Call the method
        results = manager.search("test query")
        
        # Assertions
        assert results == []
        # Verify that index was recreated
        mock_embeddings.index.assert_called_once_with([])
        mock_embeddings.save.assert_called_once()
        # Verify that embeddings class was instantiated twice (once in init, once in error handling)
        assert mock_embeddings_class.call_count == 2
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    @patch("src.embeddings.embedding_manager.time")
    def test_search_with_nonetype_error(self, mock_time, mock_embeddings_class, mock_storage_manager, clean_test_data):
        """Test searching with a 'NoneType' object has no attribute error."""
        # Set up the mocks
        mock_time.time.return_value = 12345
        
        mock_embeddings = MagicMock()
        mock_embeddings.search.side_effect = Exception("'NoneType' object has no attribute 'commit'")
        mock_embeddings_class.return_value = mock_embeddings
        
        # Create the embedding manager
        config = {
            "model": "test-model",
            "device": "cpu",
            "batch_size": 32
        }
        manager = EmbeddingManager(config, mock_storage_manager)
        
        # Create a fake index file
        index_path = mock_storage_manager.get_embeddings_path() / "index"
        index_path.parent.mkdir(exist_ok=True, parents=True)
        index_path.touch()
        
        # Call the method
        results = manager.search("test query")
        
        # Assertions
        assert results == []
        # Verify that index was recreated
        mock_embeddings.index.assert_called_once_with([])
        mock_embeddings.save.assert_called_once()
        # Verify that embeddings class was instantiated twice (once in init, once in error handling)
        assert mock_embeddings_class.call_count == 2
    
    @patch("src.embeddings.embedding_manager.Embeddings")
    def test_search_with_no_indexes_error(self, mock_embeddings_class, mock_storage_manager, clean_test_data):
        """Test searching with a 'No indexes available' error."""
        # Set up the mock
        mock_embeddings = MagicMock()
        mock_embeddings.search.side_effect = Exception("No indexes available")
        mock_embeddings_class.return_value = mock_embeddings
        
        # Create the embedding manager
        config = {
            "model": "test-model",
            "device": "cpu",
            "batch_size": 32
        }
        manager = EmbeddingManager(config, mock_storage_manager)
        
        # Create a fake index file
        index_path = mock_storage_manager.get_embeddings_path() / "index"
        index_path.parent.mkdir(exist_ok=True, parents=True)
        index_path.touch()
        
        # Call the method
        results = manager.search("test query")
        
        # Assertions
        assert results == []
        # Verify that no index recreation was attempted
        mock_embeddings.index.assert_not_called()
