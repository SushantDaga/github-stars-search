"""
Tests for the CLI utilities.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Import the module to test
from src.cli.utils import (
    load_config, save_config, get_components, get_github_api_key
)

# Sample test data
SAMPLE_CONFIG = {
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

@pytest.mark.unit
class TestLoadConfig:
    """Test the load_config function."""
    
    @patch("builtins.open", new_callable=mock_open, read_data="github:\n  per_page: 100\n")
    @patch("src.cli.utils.yaml.safe_load")
    def test_load_config(self, mock_yaml_load, mock_file):
        """Test loading configuration from a file."""
        # Set up the mock
        mock_yaml_load.return_value = {"github": {"per_page": 100}}
        
        # Call the function
        config = load_config()
        
        # Assertions
        assert config == {"github": {"per_page": 100}}
        mock_file.assert_called_once()
        mock_yaml_load.assert_called_once()
    
    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("src.cli.utils.yaml.safe_load")
    def test_load_config_file_not_found(self, mock_yaml_load, mock_file):
        """Test loading configuration when the file doesn't exist."""
        # Call the function
        config = load_config()
        
        # Assertions
        assert isinstance(config, dict)
        assert "github" in config
        assert "content" in config
        assert "embeddings" in config
        assert "search" in config
        assert "storage" in config
        
        # Check that the yaml load method was not called
        mock_yaml_load.assert_not_called()
    
    @patch("builtins.open", new_callable=mock_open, read_data="invalid: yaml: content")
    @patch("src.cli.utils.yaml.safe_load", side_effect=Exception("YAML error"))
    def test_load_config_invalid_yaml(self, mock_yaml_load, mock_file):
        """Test loading configuration with invalid YAML."""
        # Call the function
        config = load_config()
        
        # Assertions
        assert isinstance(config, dict)
        assert "github" in config
        assert "content" in config
        assert "embeddings" in config
        assert "search" in config
        assert "storage" in config

@pytest.mark.unit
class TestSaveConfig:
    """Test the save_config function."""
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.cli.utils.yaml.dump")
    def test_save_config(self, mock_yaml_dump, mock_file):
        """Test saving configuration to a file."""
        # Call the function
        result = save_config(SAMPLE_CONFIG)
        
        # Assertions
        assert result is True
        mock_file.assert_called_once()
        mock_yaml_dump.assert_called_once_with(SAMPLE_CONFIG, mock_file(), default_flow_style=False, sort_keys=False)
    
    @patch("builtins.open", side_effect=Exception("File error"))
    @patch("src.cli.utils.yaml.dump")
    def test_save_config_error(self, mock_yaml_dump, mock_file):
        """Test saving configuration with a file error."""
        # Call the function
        result = save_config(SAMPLE_CONFIG)
        
        # Assertions
        assert result is False
        mock_file.assert_called_once()
        mock_yaml_dump.assert_not_called()

@pytest.mark.unit
class TestGetComponents:
    """Test the get_components function."""
    
    @patch("src.api.github_client.GitHubClient")
    @patch("src.processor.content_processor.ContentProcessor")
    @patch("src.storage.storage_manager.StorageManager")
    @patch("src.embeddings.embedding_manager.EmbeddingManager")
    @patch("src.search.search_engine.SearchEngine")
    @patch("src.cli.utils.load_config")
    def test_get_components(self, mock_load_config, mock_search_engine, mock_embedding_manager, mock_storage_manager, mock_content_processor, mock_github_client):
        """Test getting components."""
        # Set up the mocks
        mock_load_config.return_value = SAMPLE_CONFIG
        
        mock_github_client_instance = MagicMock()
        mock_github_client.return_value = mock_github_client_instance
        
        mock_content_processor_instance = MagicMock()
        mock_content_processor.return_value = mock_content_processor_instance
        
        mock_storage_manager_instance = MagicMock()
        mock_storage_manager.return_value = mock_storage_manager_instance
        
        mock_embedding_manager_instance = MagicMock()
        mock_embedding_manager.return_value = mock_embedding_manager_instance
        
        mock_search_engine_instance = MagicMock()
        mock_search_engine.return_value = mock_search_engine_instance
        
        # Call the function
        components = get_components()
        
        # Assertions
        assert components["github_client"] == mock_github_client_instance
        assert components["content_processor"] == mock_content_processor_instance
        assert components["storage_manager"] == mock_storage_manager_instance
        assert components["embedding_manager"] == mock_embedding_manager_instance
        assert components["search_engine"] == mock_search_engine_instance
        
        # Check that the constructors were called with the correct arguments
        mock_github_client.assert_called_once_with(SAMPLE_CONFIG["github"])
        mock_content_processor.assert_called_once_with(SAMPLE_CONFIG["content"])
        mock_storage_manager.assert_called_once_with(SAMPLE_CONFIG["storage"])
        mock_embedding_manager.assert_called_once_with(SAMPLE_CONFIG["embeddings"], mock_storage_manager_instance)
        mock_search_engine.assert_called_once_with(SAMPLE_CONFIG["search"], mock_embedding_manager_instance, mock_storage_manager_instance)

@pytest.mark.unit
class TestGetGitHubApiKey:
    """Test the get_github_api_key function."""
    
    @patch.dict(os.environ, {"GITHUB_STARS_KEY": "test_api_key"})
    def test_get_github_api_key_from_env(self):
        """Test getting the GitHub API key from environment variable."""
        # Call the function
        api_key = get_github_api_key()
        
        # Assertions
        assert api_key == "test_api_key"
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("builtins.open", new_callable=mock_open, read_data="GITHUB_STARS_KEY=test_api_key")
    def test_get_github_api_key_from_dotenv(self, mock_file):
        """Test getting the GitHub API key from .env file."""
        # Call the function
        api_key = get_github_api_key()
        
        # Assertions
        assert api_key == "test_api_key"
        mock_file.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("builtins.input", return_value="test_api_key")
    def test_get_github_api_key_from_input(self, mock_input, mock_file):
        """Test getting the GitHub API key from user input."""
        # Call the function
        api_key = get_github_api_key()
        
        # Assertions
        assert api_key == "test_api_key"
        mock_file.assert_called_once()
        mock_input.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("builtins.input", return_value="")
    def test_get_github_api_key_empty_input(self, mock_input, mock_file):
        """Test getting the GitHub API key with empty user input."""
        # Call the function and check for exception
        with pytest.raises(ValueError):
            get_github_api_key()
        
        # Check that the input method was called
        mock_input.assert_called_once()
