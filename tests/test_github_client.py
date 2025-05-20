"""
Tests for the GitHub API client.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from github import GithubException

# Import the module to test
from src.api.github_client import GitHubClient

# Sample test data
SAMPLE_REPO = {
    "id": 12345,
    "name": "test-repo",
    "full_name": "test-user/test-repo",
    "description": "A test repository",
    "html_url": "https://github.com/test-user/test-repo",
    "clone_url": "https://github.com/test-user/test-repo.git",
    "language": "Python",
    "stargazers_count": 100,
    "watchers_count": 10,
    "forks_count": 20,
    "open_issues_count": 5,
    "topics": ["test", "sample"],
    "created_at": "2022-01-01T00:00:00Z",
    "updated_at": "2022-01-02T00:00:00Z",
    "pushed_at": "2022-01-03T00:00:00Z",
    "size": 1000,
    "default_branch": "main",
    "license": {"key": "mit"},
    "owner": {
        "login": "test-user",
        "id": 54321,
        "avatar_url": "https://github.com/test-user.png",
        "html_url": "https://github.com/test-user"
    }
}

SAMPLE_README = "# Test Repository\n\nThis is a test repository for unit tests."

@pytest.fixture
def mock_env_api_key(monkeypatch):
    """Set up a mock environment variable for the GitHub API key."""
    monkeypatch.setenv("GITHUB_STARS_KEY", "test_api_key")

@pytest.fixture
def github_client(mock_env_api_key):
    """Create a GitHub client instance for testing."""
    config = {"per_page": 10, "max_retries": 1, "timeout": 5}
    return GitHubClient(config)

@pytest.mark.unit
class TestGitHubClientInit:
    """Test the initialization of the GitHub client."""
    
    def test_init_with_config(self, mock_env_api_key):
        """Test initialization with configuration."""
        config = {"per_page": 20, "max_retries": 2, "timeout": 10}
        client = GitHubClient(config)
        
        assert client.per_page == 20
        assert client.max_retries == 2
        assert client.timeout == 10
        assert client.api_key == "test_api_key"
    
    def test_init_with_empty_config(self, mock_env_api_key):
        """Test initialization with empty configuration."""
        client = GitHubClient({})
        
        assert client.per_page == 100  # Default value
        assert client.max_retries == 3  # Default value
        assert client.timeout == 30  # Default value
        assert client.api_key == "test_api_key"

@pytest.mark.unit
@patch("src.api.github_client.Github")
class TestGetStarredRepositories:
    """Test the get_starred_repositories method."""
    
    def test_get_starred_repositories(self, mock_github, github_client):
        """Test getting starred repositories."""
        # Set up mock
        mock_user = MagicMock()
        mock_repo = MagicMock()
        mock_repo.id = SAMPLE_REPO["id"]
        mock_repo.name = SAMPLE_REPO["name"]
        mock_repo.full_name = SAMPLE_REPO["full_name"]
        mock_repo.description = SAMPLE_REPO["description"]
        mock_repo.html_url = SAMPLE_REPO["html_url"]
        mock_repo.clone_url = SAMPLE_REPO["clone_url"]
        mock_repo.language = SAMPLE_REPO["language"]
        mock_repo.stargazers_count = SAMPLE_REPO["stargazers_count"]
        mock_repo.watchers_count = SAMPLE_REPO["watchers_count"]
        mock_repo.forks_count = SAMPLE_REPO["forks_count"]
        mock_repo.open_issues_count = SAMPLE_REPO["open_issues_count"]
        mock_repo.topics = SAMPLE_REPO["topics"]
        # Convert string dates to datetime objects for the mock
        from datetime import datetime
        mock_repo.created_at = datetime.fromisoformat(SAMPLE_REPO["created_at"].replace("Z", "+00:00"))
        mock_repo.updated_at = datetime.fromisoformat(SAMPLE_REPO["updated_at"].replace("Z", "+00:00"))
        mock_repo.pushed_at = datetime.fromisoformat(SAMPLE_REPO["pushed_at"].replace("Z", "+00:00"))
        mock_repo.size = SAMPLE_REPO["size"]
        mock_repo.default_branch = SAMPLE_REPO["default_branch"]
        mock_repo.license = MagicMock()
        mock_repo.license.key = SAMPLE_REPO["license"]["key"]
        mock_repo.owner = MagicMock()
        mock_repo.owner.login = SAMPLE_REPO["owner"]["login"]
        mock_repo.owner.id = SAMPLE_REPO["owner"]["id"]
        mock_repo.owner.avatar_url = SAMPLE_REPO["owner"]["avatar_url"]
        mock_repo.owner.html_url = SAMPLE_REPO["owner"]["html_url"]
        
        mock_starred = MagicMock()
        mock_starred.totalCount = 1
        mock_starred.__iter__.return_value = [mock_repo]
        
        mock_user.get_starred.return_value = mock_starred
        mock_github.return_value.get_user.return_value = mock_user
        
        # Patch the github attribute to avoid authentication issues
        with patch.object(github_client, 'github', mock_github.return_value):
            # Call the method
            repos = github_client.get_starred_repositories()
            
            # Assertions
            assert len(repos) == 1
            assert repos[0]["id"] == SAMPLE_REPO["id"]
            assert repos[0]["name"] == SAMPLE_REPO["name"]
            assert repos[0]["full_name"] == SAMPLE_REPO["full_name"]
            assert repos[0]["description"] == SAMPLE_REPO["description"]
            assert repos[0]["language"] == SAMPLE_REPO["language"]
            assert repos[0]["stargazers_count"] == SAMPLE_REPO["stargazers_count"]
    
    def test_get_starred_repositories_with_limit(self, mock_github, github_client):
        """Test getting starred repositories with a limit."""
        # Set up mock
        mock_user = MagicMock()
        mock_repo1 = MagicMock()
        mock_repo1.id = 1
        mock_repo1.name = "repo1"
        mock_repo1.full_name = "user/repo1"
        mock_repo1.description = "Repository 1"
        mock_repo1.created_at = None
        mock_repo1.updated_at = None
        mock_repo1.pushed_at = None
        mock_repo1.license = None
        mock_repo1.owner = MagicMock()
        
        mock_repo2 = MagicMock()
        mock_repo2.id = 2
        mock_repo2.name = "repo2"
        mock_repo2.full_name = "user/repo2"
        mock_repo2.description = "Repository 2"
        mock_repo2.created_at = None
        mock_repo2.updated_at = None
        mock_repo2.pushed_at = None
        mock_repo2.license = None
        mock_repo2.owner = MagicMock()
        
        mock_starred = MagicMock()
        mock_starred.totalCount = 2
        mock_starred.__iter__.return_value = [mock_repo1, mock_repo2]
        
        mock_user.get_starred.return_value = mock_starred
        mock_github.return_value.get_user.return_value = mock_user
        
        # Patch the github attribute to avoid authentication issues
        with patch.object(github_client, 'github', mock_github.return_value):
            # Call the method with limit=1
            repos = github_client.get_starred_repositories(limit=1)
            
            # Assertions
            assert len(repos) == 1
            assert repos[0]["id"] == 1
            assert repos[0]["name"] == "repo1"
    
    def test_get_starred_repositories_error(self, mock_github, github_client):
        """Test error handling when getting starred repositories."""
        # Set up mock to raise an exception
        mock_user = MagicMock()
        mock_user.get_starred.side_effect = GithubException(status=500, data={"message": "API error"})
        mock_github.return_value.get_user.return_value = mock_user
        
        # Patch the github attribute to avoid authentication issues
        with patch.object(github_client, 'github', mock_github.return_value):
            # Call the method and check for exception
            with pytest.raises(GithubException):
                github_client.get_starred_repositories()

@pytest.mark.unit
@patch("src.api.github_client.Github")
class TestGetReadme:
    """Test the get_readme method."""
    
    @patch("src.api.github_client.detect")
    def test_get_readme(self, mock_detect, mock_github, github_client):
        """Test getting a repository README."""
        # Set up mock
        mock_repo = MagicMock()
        mock_readme = MagicMock()
        mock_readme.content = "IyBUZXN0IFJlcG9zaXRvcnkKClRoaXMgaXMgYSB0ZXN0IHJlcG9zaXRvcnkgZm9yIHVuaXQgdGVzdHMu"  # Base64 encoded
        
        mock_repo.get_readme.return_value = mock_readme
        mock_github.return_value.get_repo.return_value = mock_repo
        
        # Mock language detection to return English
        mock_detect.return_value = "en"
        
        # Patch the github attribute to avoid authentication issues
        with patch.object(github_client, 'github', mock_github.return_value):
            # Call the method
            readme = github_client.get_readme("test-user/test-repo")
            
            # Assertions
            assert readme == SAMPLE_README
    
    def test_get_readme_not_found(self, mock_github, github_client):
        """Test getting a README that doesn't exist."""
        # Set up mock to raise a 404 exception
        mock_github.return_value.get_repo.return_value.get_readme.side_effect = GithubException(
            status=404, data={"message": "Not Found"}
        )
        
        # Patch the github attribute to avoid authentication issues
        with patch.object(github_client, 'github', mock_github.return_value):
            # Call the method
            readme = github_client.get_readme("test-user/test-repo")
            
            # Assertions
            assert readme is None
    
    def test_get_readme_error(self, mock_github, github_client):
        """Test error handling when getting a README."""
        # Set up mock to raise a non-404 exception
        mock_github.return_value.get_repo.return_value.get_readme.side_effect = GithubException(
            status=500, data={"message": "API error"}
        )
        
        # Patch the github attribute to avoid authentication issues
        with patch.object(github_client, 'github', mock_github.return_value):
            # Call the method
            readme = github_client.get_readme("test-user/test-repo")
            
            # Assertions
            assert readme is None

@pytest.mark.unit
@patch("src.api.github_client.detect")
@patch("src.api.github_client.Github")
class TestReadmeLanguageDetection:
    """Test README language detection."""
    
    def test_english_readme(self, mock_github, mock_detect, github_client):
        """Test detecting an English README."""
        # Set up mocks
        mock_repo = MagicMock()
        mock_readme = MagicMock()
        mock_readme.content = "IyBUZXN0IFJlcG9zaXRvcnkKClRoaXMgaXMgYSB0ZXN0IHJlcG9zaXRvcnkgZm9yIHVuaXQgdGVzdHMu"  # Base64 encoded
        
        mock_repo.get_readme.return_value = mock_readme
        mock_github.return_value.get_repo.return_value = mock_repo
        
        # Mock language detection to return English
        mock_detect.return_value = "en"
        
        # Patch the github attribute to avoid authentication issues
        with patch.object(github_client, 'github', mock_github.return_value):
            # Call the method
            readme = github_client.get_readme("test-user/test-repo")
            
            # Assertions
            assert readme == SAMPLE_README
            mock_detect.assert_called_once()
    
    def test_non_english_readme(self, mock_github, mock_detect, github_client):
        """Test detecting a non-English README."""
        # Set up mocks for default README
        mock_repo = MagicMock()
        mock_readme = MagicMock()
        mock_readme.content = "IyBUZXN0IFJlcG9zaXRvcnkKClRoaXMgaXMgYSB0ZXN0IHJlcG9zaXRvcnkgZm9yIHVuaXQgdGVzdHMu"  # Base64 encoded
        
        mock_repo.get_readme.return_value = mock_readme
        mock_github.return_value.get_repo.return_value = mock_repo
        
        # Mock language detection to return non-English
        mock_detect.return_value = "fr"
        
        # Patch the github attribute to avoid authentication issues
        with patch.object(github_client, 'github', mock_github.return_value):
            # Set up mock for session.get to return None (no English README found)
            with patch.object(github_client.session, "get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 404
                mock_get.return_value = mock_response
                
                # Call the method
                readme = github_client.get_readme("test-user/test-repo")
                
                # Assertions
                assert readme == SAMPLE_README  # Should return the default README if no English version found
                mock_detect.assert_called_once()

@pytest.mark.api
@pytest.mark.slow
class TestLiveGitHubClient:
    """
    Tests that interact with the actual GitHub API.
    
    These tests are marked as slow and will be skipped by default.
    Run with: pytest -m api
    """
    
    @pytest.fixture
    def live_github_client(self):
        """Create a real GitHub client instance for testing."""
        # This will use the actual API key from the environment
        config = {"per_page": 5, "max_retries": 1, "timeout": 10}
        return GitHubClient(config)
    
    def test_get_starred_repositories_live(self, live_github_client):
        """Test getting starred repositories from the live API."""
        # Only run if GITHUB_STARS_KEY is set
        if not os.environ.get("GITHUB_STARS_KEY"):
            pytest.skip("GITHUB_STARS_KEY not set")
        
        # Get a small number of repositories
        repos = live_github_client.get_starred_repositories(limit=2)
        
        # Basic validation
        assert isinstance(repos, list)
        assert len(repos) <= 2
        
        if repos:
            # Check that the first repo has the expected fields
            repo = repos[0]
            assert "id" in repo
            assert "full_name" in repo
            assert "html_url" in repo
    
    def test_get_readme_live(self, live_github_client):
        """Test getting a README from the live API."""
        # Only run if GITHUB_STARS_KEY is set
        if not os.environ.get("GITHUB_STARS_KEY"):
            pytest.skip("GITHUB_STARS_KEY not set")
        
        # Try to get README from a well-known repository
        readme = live_github_client.get_readme("neuml/txtai")
        
        # Basic validation
        assert readme is not None
        assert "txtai" in readme.lower()
