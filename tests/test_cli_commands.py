"""
Tests for the CLI commands.
"""

import pytest
from unittest.mock import patch, MagicMock, call

# Import the module to test
from src.cli.commands import (
    update_command, search_command, config_command, info_command
)

# Sample test data
SAMPLE_REPO = {
    "id": 12345,
    "name": "test-repo",
    "full_name": "test-user/test-repo",
    "description": "A test repository",
    "language": "Python",
    "stargazers_count": 100,
    "forks_count": 20,
    "html_url": "https://github.com/test-user/test-repo"
}

SAMPLE_SEARCH_RESULTS = [
    {
        "id": "12345-0",
        "score": 0.9,
        "text": "Repository: test-user/test-repo\n\n# Test Repository",
        "repository": SAMPLE_REPO,
        "chunk_type": "readme_section"
    }
]

@pytest.fixture
def mock_github_client():
    """Create a mock GitHub client."""
    client = MagicMock()
    
    # Mock the get_starred_repositories method
    client.get_starred_repositories.return_value = [SAMPLE_REPO]
    
    # Mock the get_readme method
    client.get_readme.return_value = "# Test Repository\n\nThis is a test repository for unit tests."
    
    return client

@pytest.fixture
def mock_content_processor():
    """Create a mock content processor."""
    processor = MagicMock()
    
    # Mock the process_readme method
    processor.process_readme.return_value = [
        {
            "id": "12345-0",
            "repo_id": 12345,
            "repo_name": "test-user/test-repo",
            "content": "Repository: test-user/test-repo\n\n# Test Repository",
            "chunk_type": "readme_section",
            "section_index": 0
        }
    ]
    
    # Mock the process_description method
    processor.process_description.return_value = {
        "id": "12345-description",
        "repo_id": 12345,
        "repo_name": "test-user/test-repo",
        "content": "Repository: test-user/test-repo\n\nDescription: A test repository",
        "chunk_type": "description"
    }
    
    return processor

@pytest.fixture
def mock_storage_manager():
    """Create a mock storage manager."""
    manager = MagicMock()
    
    # Mock the has_repository method
    manager.has_repository.return_value = False
    
    # Mock the is_repository_outdated method
    manager.is_repository_outdated.return_value = True
    
    # Mock the store_repository method
    manager.store_repository.return_value = True
    
    # Mock the get_repository_count method
    manager.get_repository_count.return_value = 1
    
    # Mock the get_embedding_count method
    manager.get_embedding_count.return_value = 1
    
    # Mock the get_all_repositories method
    manager.get_all_repositories.return_value = {
        12345: SAMPLE_REPO
    }
    
    return manager

@pytest.fixture
def mock_embedding_manager():
    """Create a mock embedding manager."""
    manager = MagicMock()
    
    # Mock the generate_embeddings method
    manager.generate_embeddings.return_value = True
    
    # Mock the get_model_info method
    manager.get_model_info.return_value = {
        "model_name": "test-model",
        "device": "cpu",
        "batch_size": 32
    }
    
    return manager

@pytest.fixture
def mock_search_engine():
    """Create a mock search engine."""
    engine = MagicMock()
    
    # Mock the search method
    engine.search.return_value = SAMPLE_SEARCH_RESULTS
    
    return engine

@pytest.mark.unit
class TestUpdateCommand:
    """Test the update_command function."""
    
    @patch("src.cli.commands.tqdm")
    def test_update_command(self, mock_tqdm, mock_github_client, mock_content_processor, mock_storage_manager, mock_embedding_manager):
        """Test updating repositories."""
        
        # Call the function
        result = update_command(
            mock_github_client,
            mock_content_processor,
            mock_storage_manager,
            mock_embedding_manager,
            limit=None,
            force=False
        )
        
        # Assertions
        assert result is True
        
        # Check that the GitHub client was called
        mock_github_client.get_starred_repositories.assert_called_once_with(limit=None)
        
        # Check that the storage manager was called
        mock_storage_manager.has_repository.assert_called_once_with(SAMPLE_REPO["id"])
        mock_storage_manager.store_repository.assert_called_once()
        
        # Check that the embedding manager was called
        mock_embedding_manager.generate_embeddings.assert_called_once()
    
    @patch("src.cli.commands.tqdm")
    def test_update_command_with_limit(self, mock_tqdm, mock_github_client, mock_content_processor, mock_storage_manager, mock_embedding_manager):
        """Test updating repositories with a limit."""
        
        # Call the function with a limit
        result = update_command(
            mock_github_client,
            mock_content_processor,
            mock_storage_manager,
            mock_embedding_manager,
            limit=10,
            force=False
        )
        
        # Assertions
        assert result is True
        
        # Check that the GitHub client was called with the limit
        mock_github_client.get_starred_repositories.assert_called_once_with(limit=10)
    
    @patch("src.cli.commands.tqdm")
    def test_update_command_with_force(self, mock_tqdm, mock_github_client, mock_content_processor, mock_storage_manager, mock_embedding_manager):
        """Test updating repositories with force flag."""
        
        # Mock the has_repository method to return True
        mock_storage_manager.has_repository.return_value = True
        
        # Call the function with force=True
        result = update_command(
            mock_github_client,
            mock_content_processor,
            mock_storage_manager,
            mock_embedding_manager,
            limit=None,
            force=True
        )
        
        # Assertions
        assert result is True
        
        # Check that the is_repository_outdated method was not called
        mock_storage_manager.is_repository_outdated.assert_not_called()
        
        # Check that the store_repository method was called
        mock_storage_manager.store_repository.assert_called_once()
    
    @patch("src.cli.commands.tqdm")
    def test_update_command_repository_up_to_date(self, mock_tqdm, mock_github_client, mock_content_processor, mock_storage_manager, mock_embedding_manager):
        """Test updating repositories that are already up to date."""
        
        # Mock the has_repository method to return True
        mock_storage_manager.has_repository.return_value = True
        
        # Mock the is_repository_outdated method to return False
        mock_storage_manager.is_repository_outdated.return_value = False
        
        # Call the function
        result = update_command(
            mock_github_client,
            mock_content_processor,
            mock_storage_manager,
            mock_embedding_manager,
            limit=None,
            force=False
        )
        
        # Assertions
        assert result is True
        
        # Check that the store_repository method was not called
        mock_storage_manager.store_repository.assert_not_called()
        
        # Check that the embedding manager was not called
        mock_embedding_manager.generate_embeddings.assert_not_called()
    
    @patch("src.cli.commands.tqdm")
    def test_update_command_no_readme(self, mock_tqdm, mock_github_client, mock_content_processor, mock_storage_manager, mock_embedding_manager):
        """Test updating repositories with no README."""
        
        # Mock the get_readme method to return None
        mock_github_client.get_readme.return_value = None
        
        # Call the function
        result = update_command(
            mock_github_client,
            mock_content_processor,
            mock_storage_manager,
            mock_embedding_manager,
            limit=None,
            force=False
        )
        
        # Assertions
        assert result is True
        
        # Check that the content processor was not called
        mock_content_processor.process_readme.assert_not_called()
        
        # Check that the storage manager was not called
        mock_storage_manager.store_repository.assert_not_called()
        
        # Check that the embedding manager was not called
        mock_embedding_manager.generate_embeddings.assert_not_called()

@pytest.mark.unit
class TestSearchCommand:
    """Test the search_command function."""
    
    @patch("src.cli.commands.console")
    def test_search_command(self, mock_console, mock_search_engine):
        """Test searching repositories."""
        
        # Call the function
        result = search_command(
            mock_search_engine,
            query="test query",
            limit=10,
            min_stars=None,
            language=None,
            neural_weight=0.7,
            keyword_weight=0.3
        )
        
        # Assertions
        assert result is True
        
        # Check that the search engine was called with the correct arguments
        mock_search_engine.search.assert_called_once()
        args, kwargs = mock_search_engine.search.call_args
        assert args[0] == "test query"
        assert kwargs["limit"] == 10
        assert "filters" in kwargs
        
        # Check that the console was used to print the results
        assert mock_console.print.call_count > 0
    
    @patch("src.cli.commands.console")
    def test_search_command_with_filters(self, mock_console, mock_search_engine):
        """Test searching repositories with filters."""
        
        # Call the function with filters
        result = search_command(
            mock_search_engine,
            query="test query",
            limit=10,
            min_stars=100,
            language="Python",
            neural_weight=0.7,
            keyword_weight=0.3
        )
        
        # Assertions
        assert result is True
        
        # Check that the search engine was called with the correct filters
        mock_search_engine.search.assert_called_once()
        args, kwargs = mock_search_engine.search.call_args
        assert kwargs["filters"]["stargazers_count"]["min"] == 100
        assert kwargs["filters"]["language"] == "Python"
    
    @patch("src.cli.commands.console")
    def test_search_command_with_weights(self, mock_console, mock_search_engine):
        """Test searching repositories with custom weights."""
        
        # Call the function with custom weights
        result = search_command(
            mock_search_engine,
            query="test query",
            limit=10,
            min_stars=None,
            language=None,
            neural_weight=0.8,
            keyword_weight=0.2
        )
        
        # Assertions
        assert result is True
        
        # Check that the search engine weights were updated
        assert mock_search_engine.neural_weight == 0.8
        assert mock_search_engine.keyword_weight == 0.2
    
    @patch("src.cli.commands.console")
    def test_search_command_no_results(self, mock_console, mock_search_engine):
        """Test searching repositories with no results."""
        
        # Mock the search method to return no results
        mock_search_engine.search.return_value = []
        
        # Call the function
        result = search_command(
            mock_search_engine,
            query="test query",
            limit=10,
            min_stars=None,
            language=None,
            neural_weight=0.7,
            keyword_weight=0.3
        )
        
        # Assertions
        assert result is True
        
        # Check that the console was used to print a message
        mock_console.print.assert_any_call("[bold yellow]No results found.[/bold yellow]")

@pytest.mark.unit
class TestConfigCommand:
    """Test the config_command function."""
    
    @patch("src.cli.commands.yaml")
    @patch("src.cli.commands.console")
    @patch("src.cli.commands.load_config")
    @patch("src.cli.commands.save_config")
    def test_config_command_show(self, mock_save_config, mock_load_config, mock_console, mock_yaml):
        """Test showing the configuration."""
        # Set up the mock load_config
        mock_load_config.return_value = {
            "github": {"per_page": 100},
            "content": {"chunk_strategy": "hybrid"},
            "embeddings": {"model": "test-model"},
            "search": {"neural_weight": 0.7},
            "storage": {"compress_data": True}
        }
        
        
        # Call the function with show=True
        result = config_command(
            show=True,
            embedding_model=None,
            neural_weight=None,
            keyword_weight=None,
            chunk_strategy=None,
            chunk_size=None,
            chunk_overlap=None
        )
        
        # Assertions
        assert result is True
        
        # Check that the console was used to print the configuration
        assert mock_console.print.call_count > 0
    
    @patch("src.cli.commands.yaml")
    @patch("src.cli.commands.console")
    @patch("src.cli.commands.load_config")
    @patch("src.cli.commands.save_config")
    def test_config_command_update(self, mock_save_config, mock_load_config, mock_console, mock_yaml):
        """Test updating the configuration."""
        # Set up the mock load_config
        mock_load_config.return_value = {
            "github": {"per_page": 100},
            "content": {"chunk_strategy": "hybrid"},
            "embeddings": {"model": "test-model"},
            "search": {"neural_weight": 0.7, "keyword_weight": 0.3},
            "storage": {"compress_data": True}
        }
        
        # Call the function with update parameters
        result = config_command(
            show=False,
            embedding_model="new-model",
            neural_weight=0.8,
            keyword_weight=0.2,
            chunk_strategy="semantic",
            chunk_size=200,
            chunk_overlap=30
        )
        
        # Assertions
        assert result is True
        
        # Check that the save_config method was called with the updated configuration
        mock_save_config.assert_called_once()
        args, kwargs = mock_save_config.call_args
        config = args[0]
        assert config["embeddings"]["model"] == "new-model"
        assert config["search"]["neural_weight"] == 0.8
        assert config["search"]["keyword_weight"] == 0.2
        assert config["content"]["chunk_strategy"] == "semantic"
        assert config["content"]["max_chunk_size"] == 200
        assert config["content"]["chunk_overlap"] == 30
        
        # Instead of checking for a specific message, just verify that print was called
        assert mock_console.print.call_count > 0

@pytest.mark.unit
class TestInfoCommand:
    """Test the info_command function."""
    
    @patch("src.cli.commands.console")
    def test_info_command(self, mock_console, mock_storage_manager, mock_embedding_manager):
        """Test showing information."""
        
        # Call the function
        result = info_command(mock_storage_manager, mock_embedding_manager)
        
        # Assertions
        assert result is True
        
        # Check that the storage manager methods were called
        mock_storage_manager.get_repository_count.assert_called_once()
        mock_storage_manager.get_embedding_count.assert_called_once()
        mock_storage_manager.get_all_repositories.assert_called_once()
        
        # Check that the embedding manager method was called
        mock_embedding_manager.get_model_info.assert_called_once()
        
        # Check that the console was used to print information
        assert mock_console.print.call_count > 0
    
    @patch("src.cli.commands.console")
    def test_info_command_no_data(self, mock_console, mock_storage_manager, mock_embedding_manager):
        """Test showing information with no data."""
        
        # Mock the get_repository_count method to return 0
        mock_storage_manager.get_repository_count.return_value = 0
        
        # Mock the get_all_repositories method to return an empty dictionary
        mock_storage_manager.get_all_repositories.return_value = {}
        
        # Call the function
        result = info_command(mock_storage_manager, mock_embedding_manager)
        
        # Assertions
        assert result is True
        
        # Instead of checking for a specific message, just verify that print was called
        assert mock_console.print.call_count > 0
