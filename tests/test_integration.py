"""
Integration tests for GitHub Stars Search.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules to test
from src.api.github_client import GitHubClient
from src.processor.content_processor import ContentProcessor
from src.embeddings.embedding_manager import EmbeddingManager
from src.search.search_engine import SearchEngine
from src.storage.storage_manager import StorageManager

# Sample test data
SAMPLE_REPO = {
    "id": 12345,
    "name": "test-repo",
    "full_name": "test-user/test-repo",
    "description": "A test repository",
    "language": "Python",
    "stargazers_count": 100,
    "forks_count": 20,
    "html_url": "https://github.com/test-user/test-repo",
    "updated_at": "2022-01-01T00:00:00Z"
}

SAMPLE_README = """
# Test Repository

This is a test repository for integration tests.

## Section 1

This is the first section of the README.

## Section 2

This is the second section of the README.
"""

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_github_client():
    """Create a mock GitHub client."""
    client = MagicMock()
    
    # Mock the get_starred_repositories method
    client.get_starred_repositories.return_value = [SAMPLE_REPO]
    
    # Mock the get_readme method
    client.get_readme.return_value = SAMPLE_README
    
    return client

@pytest.mark.integration
class TestComponentIntegration:
    """Test the integration between components."""
    
    def test_update_and_search_flow(self, temp_dir, mock_github_client):
        """Test the update and search flow."""
        # Create the components
        storage_config = {
            "compress_data": False,
            "backup_enabled": False,
            "max_backups": 1
        }
        
        content_config = {
            "max_readme_size": 10000,
            "chunk_strategy": "hybrid",
            "max_chunk_size": 100,
            "chunk_overlap": 20
        }
        
        embeddings_config = {
            "model": "BAAI/bge-small-en-v1.5",
            "device": "cpu",
            "batch_size": 32,
            "cache_enabled": True
        }
        
        search_config = {
            "hybrid_enabled": True,
            "neural_weight": 0.7,
            "keyword_weight": 0.3,
            "max_results": 10,
            "min_score": 0.2
        }
        
        # Create the storage manager with a temporary directory
        storage_manager = StorageManager(storage_config)
        
        # Override the paths to use the temp directory
        storage_manager.base_path = Path(temp_dir)
        storage_manager.data_path = Path(temp_dir) / "data"
        storage_manager.repositories_path = Path(temp_dir) / "data" / "repositories"
        storage_manager.embeddings_path = Path(temp_dir) / "data" / "embeddings"
        storage_manager.index_path = Path(temp_dir) / "data" / "index"
        
        # Create the directories
        storage_manager.repositories_path.mkdir(parents=True, exist_ok=True)
        storage_manager.embeddings_path.mkdir(parents=True, exist_ok=True)
        storage_manager.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the repository index
        storage_manager.repository_index = {}
        
        # Create the content processor
        content_processor = ContentProcessor(content_config)
        
        # Create the embedding manager with a mock embeddings instance
        with patch("src.embeddings.embedding_manager.Embeddings") as mock_embeddings_class:
            # Set up the mock
            mock_embeddings_instance = MagicMock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            
            # Mock the search method
            mock_embeddings_instance.search.return_value = [
                {
                    "id": "12345-0",
                    "text": "Repository: test-user/test-repo\n\n# Test Repository",
                    "score": 0.9,
                    "repo_id": 12345,
                    "repo_name": "test-user/test-repo",
                    "chunk_type": "readme_section"
                }
            ]
            
            # Create the embedding manager
            embedding_manager = EmbeddingManager(embeddings_config, storage_manager)
            
            # Create the search engine
            search_engine = SearchEngine(search_config, embedding_manager, storage_manager)
            
            # Step 1: Process the repository
            readme_chunks = content_processor.process_readme(SAMPLE_README, SAMPLE_REPO)
            
            # Step 2: Store the repository
            storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, readme_chunks)
            
            # Step 3: Generate embeddings
            embedding_manager.generate_embeddings(SAMPLE_REPO["id"], readme_chunks)
            
            # Step 4: Search for content
            results = search_engine.search("test repository")
            
            # Assertions
            assert len(results) == 1
            assert results[0]["repository"]["id"] == SAMPLE_REPO["id"]
            assert results[0]["repository"]["full_name"] == SAMPLE_REPO["full_name"]
            assert results[0]["score"] > 0
    
    def test_end_to_end_update_flow(self, temp_dir, mock_github_client):
        """Test the end-to-end update flow."""
        # Create the components
        storage_config = {
            "compress_data": False,
            "backup_enabled": False,
            "max_backups": 1
        }
        
        content_config = {
            "max_readme_size": 10000,
            "chunk_strategy": "hybrid",
            "max_chunk_size": 100,
            "chunk_overlap": 20
        }
        
        embeddings_config = {
            "model": "BAAI/bge-small-en-v1.5",
            "device": "cpu",
            "batch_size": 32,
            "cache_enabled": True
        }
        
        # Create the storage manager with a temporary directory
        storage_manager = StorageManager(storage_config)
        
        # Override the paths to use the temp directory
        storage_manager.base_path = Path(temp_dir)
        storage_manager.data_path = Path(temp_dir) / "data"
        storage_manager.repositories_path = Path(temp_dir) / "data" / "repositories"
        storage_manager.embeddings_path = Path(temp_dir) / "data" / "embeddings"
        storage_manager.index_path = Path(temp_dir) / "data" / "index"
        
        # Create the directories
        storage_manager.repositories_path.mkdir(parents=True, exist_ok=True)
        storage_manager.embeddings_path.mkdir(parents=True, exist_ok=True)
        storage_manager.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the repository index
        storage_manager.repository_index = {}
        
        # Create the content processor
        content_processor = ContentProcessor(content_config)
        
        # Create the embedding manager with a mock embeddings instance
        with patch("src.embeddings.embedding_manager.Embeddings") as mock_embeddings_class:
            # Set up the mock
            mock_embeddings_instance = MagicMock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            
            # Create the embedding manager
            embedding_manager = EmbeddingManager(embeddings_config, storage_manager)
            
            # Import the update command
            from src.cli.commands import update_command
            
            # Run the update command
            result = update_command(
                mock_github_client,
                content_processor,
                storage_manager,
                embedding_manager,
                limit=None,
                force=False
            )
            
            # Assertions
            assert result is True
            
            # Check that the repository was stored
            assert storage_manager.has_repository(SAMPLE_REPO["id"])
            
            # Check that the repository data was stored correctly
            repo = storage_manager.get_repository(SAMPLE_REPO["id"])
            assert repo["id"] == SAMPLE_REPO["id"]
            assert repo["full_name"] == SAMPLE_REPO["full_name"]
            
            # Check that the README was stored
            readme = storage_manager.get_repository_readme(SAMPLE_REPO["id"])
            assert readme == SAMPLE_README
            
            # Check that the chunks were stored
            chunks = storage_manager.get_repository_chunks(SAMPLE_REPO["id"])
            assert len(chunks) > 0
            
            # Check that the repository was marked as embedded
            assert storage_manager.has_embeddings(SAMPLE_REPO["id"])

@pytest.mark.api
@pytest.mark.slow
class TestLiveApiIntegration:
    """
    Tests that interact with the actual GitHub API.
    
    These tests are marked as slow and will be skipped by default.
    Run with: pytest -m api
    """
    
    def test_live_github_client_integration(self, temp_dir):
        """Test integration with the live GitHub API."""
        # Only run if GITHUB_STARS_KEY is set
        if not os.environ.get("GITHUB_STARS_KEY"):
            pytest.skip("GITHUB_STARS_KEY not set")
        
        # Create the components
        github_config = {
            "per_page": 5,
            "max_retries": 1,
            "timeout": 10
        }
        
        storage_config = {
            "compress_data": False,
            "backup_enabled": False,
            "max_backups": 1
        }
        
        content_config = {
            "max_readme_size": 10000,
            "chunk_strategy": "hybrid",
            "max_chunk_size": 100,
            "chunk_overlap": 20
        }
        
        # Create the GitHub client
        github_client = GitHubClient(github_config)
        
        # Create the storage manager with a temporary directory
        storage_manager = StorageManager(storage_config)
        
        # Override the paths to use the temp directory
        storage_manager.base_path = Path(temp_dir)
        storage_manager.data_path = Path(temp_dir) / "data"
        storage_manager.repositories_path = Path(temp_dir) / "data" / "repositories"
        storage_manager.embeddings_path = Path(temp_dir) / "data" / "embeddings"
        storage_manager.index_path = Path(temp_dir) / "data" / "index"
        
        # Create the directories
        storage_manager.repositories_path.mkdir(parents=True, exist_ok=True)
        storage_manager.embeddings_path.mkdir(parents=True, exist_ok=True)
        storage_manager.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the repository index
        storage_manager.repository_index = {}
        
        # Create the content processor
        content_processor = ContentProcessor(content_config)
        
        # Get a small number of repositories
        repositories = github_client.get_starred_repositories(limit=1)
        
        # Basic validation
        assert isinstance(repositories, list)
        assert len(repositories) <= 1
        
        if repositories:
            # Get the first repository
            repo = repositories[0]
            
            # Get the README
            readme = github_client.get_readme(repo["full_name"])
            
            # Process the README
            if readme:
                chunks = content_processor.process_readme(readme, repo)
                
                # Store the repository
                storage_manager.store_repository(repo, readme, chunks)
                
                # Check that the repository was stored
                assert storage_manager.has_repository(repo["id"])
                
                # Check that the repository data was stored correctly
                stored_repo = storage_manager.get_repository(repo["id"])
                assert stored_repo["id"] == repo["id"]
                assert stored_repo["full_name"] == repo["full_name"]
