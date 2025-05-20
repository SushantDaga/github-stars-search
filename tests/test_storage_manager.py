"""
Tests for the storage manager.
"""

import os
import json
import shutil
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module to test
from src.storage.storage_manager import StorageManager

# Sample test data
SAMPLE_REPO = {
    "id": 12345,
    "name": "test-repo",
    "full_name": "test-user/test-repo",
    "description": "A test repository",
    "updated_at": "2022-01-01T00:00:00Z"
}

SAMPLE_README = "# Test Repository\n\nThis is a test repository for unit tests."

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
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def storage_manager(temp_dir):
    """Create a storage manager instance for testing."""
    config = {
        "compress_data": False,
        "backup_enabled": False,
        "max_backups": 1
    }
    
    # Create a storage manager with the base path set to the temp directory
    manager = StorageManager(config)
    
    # Override the paths to use the temp directory
    manager.base_path = Path(temp_dir)
    manager.data_path = Path(temp_dir) / "data"
    manager.repositories_path = Path(temp_dir) / "data" / "repositories"
    manager.embeddings_path = Path(temp_dir) / "data" / "embeddings"
    manager.index_path = Path(temp_dir) / "data" / "index"
    
    # Create the directories
    manager.repositories_path.mkdir(parents=True, exist_ok=True)
    manager.embeddings_path.mkdir(parents=True, exist_ok=True)
    manager.index_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize the repository index
    manager.repository_index = {}
    
    return manager

@pytest.mark.unit
class TestStorageManagerInit:
    """Test the initialization of the storage manager."""
    
    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = {
            "compress_data": True,
            "backup_enabled": True,
            "max_backups": 5
        }
        manager = StorageManager(config)
        
        assert manager.compress_data is True
        assert manager.backup_enabled is True
        assert manager.max_backups == 5
    
    def test_init_with_empty_config(self):
        """Test initialization with empty configuration."""
        manager = StorageManager({})
        
        assert manager.compress_data is True  # Default value
        assert manager.backup_enabled is True  # Default value
        assert manager.max_backups == 3  # Default value

@pytest.mark.unit
class TestRepositoryIndex:
    """Test the repository index functionality."""
    
    def test_load_repository_index_empty(self, storage_manager):
        """Test loading an empty repository index."""
        # Call the method
        index = storage_manager._load_repository_index()
        
        # Assertions
        assert index == {}
    
    def test_load_repository_index(self, storage_manager):
        """Test loading a repository index."""
        # Create a sample index file
        index_data = {
            "12345": {
                "id": 12345,
                "full_name": "test-user/test-repo",
                "updated_at": "2022-01-01T00:00:00Z",
                "embedded": True,
                "stored_at": "2022-01-02T00:00:00Z"
            }
        }
        
        index_file = storage_manager.index_path / "repositories.json"
        with open(index_file, "w") as f:
            json.dump(index_data, f)
        
        # Call the method
        index = storage_manager._load_repository_index()
        
        # Assertions
        assert index == index_data
    
    def test_save_repository_index(self, storage_manager):
        """Test saving the repository index."""
        # Set up the index
        storage_manager.repository_index = {
            "12345": {
                "id": 12345,
                "full_name": "test-user/test-repo",
                "updated_at": "2022-01-01T00:00:00Z",
                "embedded": True,
                "stored_at": "2022-01-02T00:00:00Z"
            }
        }
        
        # Call the method
        storage_manager._save_repository_index()
        
        # Check that the file was created
        index_file = storage_manager.index_path / "repositories.json"
        assert index_file.exists()
        
        # Check the content
        with open(index_file, "r") as f:
            saved_index = json.load(f)
        
        assert saved_index == storage_manager.repository_index

@pytest.mark.unit
class TestStoreRepository:
    """Test the store_repository method."""
    
    def test_store_repository(self, storage_manager):
        """Test storing a repository."""
        # Call the method
        result = storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Assertions
        assert result is True
        
        # Check that the repository directory was created
        repo_dir = storage_manager.repositories_path / str(SAMPLE_REPO["id"])
        assert repo_dir.exists()
        
        # Check that the files were created
        metadata_file = repo_dir / "metadata.json"
        readme_file = repo_dir / "readme.md"
        chunks_file = repo_dir / "chunks.json"
        
        assert metadata_file.exists()
        assert readme_file.exists()
        assert chunks_file.exists()
        
        # Check the content of the files
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        with open(readme_file, "r") as f:
            readme = f.read()
        
        with open(chunks_file, "r") as f:
            chunks = json.load(f)
        
        assert metadata == SAMPLE_REPO
        assert readme == SAMPLE_README
        assert chunks == SAMPLE_CHUNKS
        
        # Check that the repository was added to the index
        assert str(SAMPLE_REPO["id"]) in storage_manager.repository_index
        assert storage_manager.repository_index[str(SAMPLE_REPO["id"])]["full_name"] == SAMPLE_REPO["full_name"]
        assert storage_manager.repository_index[str(SAMPLE_REPO["id"])]["updated_at"] == SAMPLE_REPO["updated_at"]
        assert storage_manager.repository_index[str(SAMPLE_REPO["id"])]["embedded"] is False
    
    def test_store_repository_with_backup(self, storage_manager):
        """Test storing a repository with backup enabled."""
        # Enable backups
        storage_manager.backup_enabled = True
        
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Mock the backup method
        with patch.object(storage_manager, '_backup_repository') as mock_backup:
            # Store the repository again
            storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
            
            # Check that the backup method was called
            mock_backup.assert_called_once_with(SAMPLE_REPO["id"])

@pytest.mark.unit
class TestGetRepository:
    """Test the get_repository method."""
    
    def test_get_repository(self, storage_manager):
        """Test getting a repository."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Call the method
        repo = storage_manager.get_repository(SAMPLE_REPO["id"])
        
        # Assertions
        assert repo == SAMPLE_REPO
    
    def test_get_repository_not_found(self, storage_manager):
        """Test getting a repository that doesn't exist."""
        # Call the method
        repo = storage_manager.get_repository(99999)
        
        # Assertions
        assert repo is None

@pytest.mark.unit
class TestGetRepositoryReadme:
    """Test the get_repository_readme method."""
    
    def test_get_repository_readme(self, storage_manager):
        """Test getting a repository README."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Call the method
        readme = storage_manager.get_repository_readme(SAMPLE_REPO["id"])
        
        # Assertions
        assert readme == SAMPLE_README
    
    def test_get_repository_readme_not_found(self, storage_manager):
        """Test getting a README for a repository that doesn't exist."""
        # Call the method
        readme = storage_manager.get_repository_readme(99999)
        
        # Assertions
        assert readme is None

@pytest.mark.unit
class TestGetRepositoryChunks:
    """Test the get_repository_chunks method."""
    
    def test_get_repository_chunks(self, storage_manager):
        """Test getting repository chunks."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Call the method
        chunks = storage_manager.get_repository_chunks(SAMPLE_REPO["id"])
        
        # Assertions
        assert chunks == SAMPLE_CHUNKS
    
    def test_get_repository_chunks_not_found(self, storage_manager):
        """Test getting chunks for a repository that doesn't exist."""
        # Call the method
        chunks = storage_manager.get_repository_chunks(99999)
        
        # Assertions
        assert chunks is None

@pytest.mark.unit
class TestHasRepository:
    """Test the has_repository method."""
    
    def test_has_repository(self, storage_manager):
        """Test checking if a repository exists."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Call the method
        has_repo = storage_manager.has_repository(SAMPLE_REPO["id"])
        
        # Assertions
        assert has_repo is True
    
    def test_has_repository_not_found(self, storage_manager):
        """Test checking if a repository that doesn't exist exists."""
        # Call the method
        has_repo = storage_manager.has_repository(99999)
        
        # Assertions
        assert has_repo is False

@pytest.mark.unit
class TestIsRepositoryOutdated:
    """Test the is_repository_outdated method."""
    
    def test_is_repository_outdated_same_date(self, storage_manager):
        """Test checking if a repository is outdated with the same date."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Call the method with the same repository
        is_outdated = storage_manager.is_repository_outdated(SAMPLE_REPO)
        
        # Assertions
        assert is_outdated is False
    
    def test_is_repository_outdated_different_date(self, storage_manager):
        """Test checking if a repository is outdated with a different date."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Create a repository with a different updated_at
        updated_repo = SAMPLE_REPO.copy()
        updated_repo["updated_at"] = "2022-01-02T00:00:00Z"
        
        # Call the method with the updated repository
        is_outdated = storage_manager.is_repository_outdated(updated_repo)
        
        # Assertions
        assert is_outdated is True
    
    def test_is_repository_outdated_not_found(self, storage_manager):
        """Test checking if a repository that doesn't exist is outdated."""
        # Create a repository with a different ID
        different_repo = SAMPLE_REPO.copy()
        different_repo["id"] = 99999
        
        # Call the method with the different repository
        is_outdated = storage_manager.is_repository_outdated(different_repo)
        
        # Assertions
        assert is_outdated is True

@pytest.mark.unit
class TestMarkRepositoryEmbedded:
    """Test the mark_repository_embedded method."""
    
    def test_mark_repository_embedded(self, storage_manager):
        """Test marking a repository as embedded."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Call the method
        storage_manager.mark_repository_embedded(SAMPLE_REPO["id"])
        
        # Assertions
        assert storage_manager.repository_index[str(SAMPLE_REPO["id"])]["embedded"] is True
    
    def test_mark_repository_embedded_not_found(self, storage_manager):
        """Test marking a repository that doesn't exist as embedded."""
        # Call the method
        storage_manager.mark_repository_embedded(99999)
        
        # Assertions
        assert "99999" not in storage_manager.repository_index

@pytest.mark.unit
class TestHasEmbeddings:
    """Test the has_embeddings method."""
    
    def test_has_embeddings(self, storage_manager):
        """Test checking if a repository has embeddings."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Mark the repository as embedded
        storage_manager.mark_repository_embedded(SAMPLE_REPO["id"])
        
        # Call the method
        has_embeddings = storage_manager.has_embeddings(SAMPLE_REPO["id"])
        
        # Assertions
        assert has_embeddings is True
    
    def test_has_embeddings_not_embedded(self, storage_manager):
        """Test checking if a repository that isn't embedded has embeddings."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Call the method
        has_embeddings = storage_manager.has_embeddings(SAMPLE_REPO["id"])
        
        # Assertions
        assert has_embeddings is False
    
    def test_has_embeddings_not_found(self, storage_manager):
        """Test checking if a repository that doesn't exist has embeddings."""
        # Call the method
        has_embeddings = storage_manager.has_embeddings(99999)
        
        # Assertions
        assert has_embeddings is False

@pytest.mark.unit
class TestGetEmbeddingsPath:
    """Test the get_embeddings_path method."""
    
    def test_get_embeddings_path(self, storage_manager):
        """Test getting the embeddings path."""
        # Call the method
        path = storage_manager.get_embeddings_path()
        
        # Assertions
        assert path == storage_manager.embeddings_path

@pytest.mark.unit
class TestGetAllRepositories:
    """Test the get_all_repositories method."""
    
    def test_get_all_repositories(self, storage_manager):
        """Test getting all repositories."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Call the method
        repos = storage_manager.get_all_repositories()
        
        # Assertions
        assert len(repos) == 1
        assert repos[SAMPLE_REPO["id"]] == SAMPLE_REPO
    
    def test_get_all_repositories_empty(self, storage_manager):
        """Test getting all repositories when there are none."""
        # Call the method
        repos = storage_manager.get_all_repositories()
        
        # Assertions
        assert repos == {}

@pytest.mark.unit
class TestGetRepositoryCount:
    """Test the get_repository_count method."""
    
    def test_get_repository_count(self, storage_manager):
        """Test getting the repository count."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Call the method
        count = storage_manager.get_repository_count()
        
        # Assertions
        assert count == 1
    
    def test_get_repository_count_empty(self, storage_manager):
        """Test getting the repository count when there are none."""
        # Call the method
        count = storage_manager.get_repository_count()
        
        # Assertions
        assert count == 0

@pytest.mark.unit
class TestGetEmbeddingCount:
    """Test the get_embedding_count method."""
    
    def test_get_embedding_count(self, storage_manager):
        """Test getting the embedding count."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Mark the repository as embedded
        storage_manager.mark_repository_embedded(SAMPLE_REPO["id"])
        
        # Call the method
        count = storage_manager.get_embedding_count()
        
        # Assertions
        assert count == 1
    
    def test_get_embedding_count_not_embedded(self, storage_manager):
        """Test getting the embedding count when no repositories are embedded."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Call the method
        count = storage_manager.get_embedding_count()
        
        # Assertions
        assert count == 0
    
    def test_get_embedding_count_empty(self, storage_manager):
        """Test getting the embedding count when there are no repositories."""
        # Call the method
        count = storage_manager.get_embedding_count()
        
        # Assertions
        assert count == 0

@pytest.mark.unit
class TestHasData:
    """Test the has_data method."""
    
    def test_has_data(self, storage_manager):
        """Test checking if there is any data."""
        # First store the repository
        storage_manager.store_repository(SAMPLE_REPO, SAMPLE_README, SAMPLE_CHUNKS)
        
        # Call the method
        has_data = storage_manager.has_data()
        
        # Assertions
        assert has_data is True
    
    def test_has_data_empty(self, storage_manager):
        """Test checking if there is any data when there is none."""
        # Call the method
        has_data = storage_manager.has_data()
        
        # Assertions
        assert has_data is False
