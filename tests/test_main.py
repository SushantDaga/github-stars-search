"""
Tests for the main script.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import the module to test
import github_stars_search

@pytest.mark.unit
class TestMainScript:
    """Test the main script."""
    
    @patch("github_stars_search.get_components")
    @patch("github_stars_search.update_command")
    def test_update_command(self, mock_update_command, mock_get_components):
        """Test the update command."""
        # Set up the mocks
        mock_get_components.return_value = {
            "github_client": MagicMock(),
            "content_processor": MagicMock(),
            "storage_manager": MagicMock(),
            "embedding_manager": MagicMock()
        }
        mock_update_command.return_value = True
        
        # Call the function with update command
        with patch("sys.argv", ["github_stars_search.py", "update"]):
            with patch("sys.exit") as mock_exit:  # Patch sys.exit to prevent test from exiting
                github_stars_search.main()
        
        # Assertions
        mock_get_components.assert_called_once()
        mock_update_command.assert_called_once()
    
    @patch("github_stars_search.get_components")
    @patch("github_stars_search.update_command")
    def test_update_command_with_args(self, mock_update_command, mock_get_components):
        """Test the update command with arguments."""
        # Set up the mocks
        mock_get_components.return_value = {
            "github_client": MagicMock(),
            "content_processor": MagicMock(),
            "storage_manager": MagicMock(),
            "embedding_manager": MagicMock()
        }
        mock_update_command.return_value = True
        
        # Call the function with update command and arguments
        with patch("sys.argv", ["github_stars_search.py", "update", "--limit", "10", "--force"]):
            with patch("sys.exit") as mock_exit:  # Patch sys.exit to prevent test from exiting
                github_stars_search.main()
        
        # Assertions
        mock_get_components.assert_called_once()
        mock_update_command.assert_called_once()
        args, kwargs = mock_update_command.call_args
        assert kwargs["limit"] == 10
        assert kwargs["force"] is True
    
    @patch("github_stars_search.get_components")
    @patch("github_stars_search.search_command")
    def test_search_command(self, mock_search_command, mock_get_components):
        """Test the search command."""
        # Set up the mocks
        mock_get_components.return_value = {
            "search_engine": MagicMock()
        }
        mock_search_command.return_value = True
        
        # Call the function with search command
        with patch("sys.argv", ["github_stars_search.py", "search", "test query"]):
            with patch("sys.exit") as mock_exit:  # Patch sys.exit to prevent test from exiting
                github_stars_search.main()
        
        # Assertions
        mock_get_components.assert_called_once()
        mock_search_command.assert_called_once()
    
    @patch("github_stars_search.get_components")
    @patch("github_stars_search.search_command")
    def test_search_command_with_args(self, mock_search_command, mock_get_components):
        """Test the search command with arguments."""
        # Set up the mocks
        mock_get_components.return_value = {
            "search_engine": MagicMock()
        }
        mock_search_command.return_value = True
        
        # Call the function with search command and arguments
        with patch("sys.argv", ["github_stars_search.py", "search", "test query", "--limit", "10", "--min-stars", "100", "--language", "Python", "--neural-weight", "0.8", "--keyword-weight", "0.2"]):
            with patch("sys.exit") as mock_exit:  # Patch sys.exit to prevent test from exiting
                github_stars_search.main()
        
        # Assertions
        mock_get_components.assert_called_once()
        mock_search_command.assert_called_once()
        args, kwargs = mock_search_command.call_args
        assert kwargs["query"] == "test query"
        assert kwargs["limit"] == 10
        assert kwargs["min_stars"] == 100
        assert kwargs["language"] == "Python"
        assert kwargs["neural_weight"] == 0.8
        assert kwargs["keyword_weight"] == 0.2
    
    @patch("github_stars_search.config_command")
    def test_config_command(self, mock_config_command):
        """Test the config command."""
        # Set up the mock
        mock_config_command.return_value = True
        
        # Call the function with config command
        with patch("sys.argv", ["github_stars_search.py", "config", "--show"]):
            with patch("sys.exit") as mock_exit:  # Patch sys.exit to prevent test from exiting
                github_stars_search.main()
        
        # Assertions
        mock_config_command.assert_called_once()
        args, kwargs = mock_config_command.call_args
        assert kwargs["show"] is True
    
    @patch("github_stars_search.config_command")
    def test_config_command_with_args(self, mock_config_command):
        """Test the config command with arguments."""
        # Set up the mock
        mock_config_command.return_value = True
        
        # Call the function with config command and arguments
        with patch("sys.argv", ["github_stars_search.py", "config", "--embedding-model", "new-model", "--neural-weight", "0.8", "--keyword-weight", "0.2", "--chunk-strategy", "semantic", "--chunk-size", "200", "--chunk-overlap", "30"]):
            with patch("sys.exit") as mock_exit:  # Patch sys.exit to prevent test from exiting
                github_stars_search.main()
        
        # Assertions
        mock_config_command.assert_called_once()
        args, kwargs = mock_config_command.call_args
        assert kwargs["show"] is False
        assert kwargs["embedding_model"] == "new-model"
        assert kwargs["neural_weight"] == 0.8
        assert kwargs["keyword_weight"] == 0.2
        assert kwargs["chunk_strategy"] == "semantic"
        assert kwargs["chunk_size"] == 200
        assert kwargs["chunk_overlap"] == 30
    
    @patch("github_stars_search.get_components")
    @patch("github_stars_search.info_command")
    def test_info_command(self, mock_info_command, mock_get_components):
        """Test the info command."""
        # Set up the mocks
        mock_get_components.return_value = {
            "storage_manager": MagicMock(),
            "embedding_manager": MagicMock()
        }
        mock_info_command.return_value = True
        
        # Call the function with info command
        with patch("sys.argv", ["github_stars_search.py", "info"]):
            with patch("sys.exit") as mock_exit:  # Patch sys.exit to prevent test from exiting
                github_stars_search.main()
        
        # Assertions
        mock_get_components.assert_called_once()
        mock_info_command.assert_called_once()
    
    def test_invalid_command(self):
        """Test an invalid command."""
        # Call the function with an invalid command
        with patch("sys.argv", ["github_stars_search.py", "invalid"]):
            with patch("sys.exit") as mock_exit:  # Patch sys.exit to prevent test from exiting
                # We're just testing that it doesn't crash
                github_stars_search.main()
    
    def test_no_command(self):
        """Test no command."""
        # Call the function with no command
        with patch("sys.argv", ["github_stars_search.py"]):
            with patch("sys.exit") as mock_exit:  # Patch sys.exit to prevent test from exiting
                # We're just testing that it doesn't crash
                github_stars_search.main()
    
    @patch("github_stars_search.get_components")
    @patch("github_stars_search.update_command", side_effect=Exception("Test error"))
    @patch("github_stars_search.console.print")  # Patch the rich console.print instead of builtins.print
    def test_command_error(self, mock_console_print, mock_update_command, mock_get_components):
        """Test error handling in commands."""
        # Set up the mocks
        mock_get_components.return_value = {
            "github_client": MagicMock(),
            "content_processor": MagicMock(),
            "storage_manager": MagicMock(),
            "embedding_manager": MagicMock()
        }
        
        # Call the function with update command
        with patch("sys.argv", ["github_stars_search.py", "update"]):
            with patch("sys.exit") as mock_exit:  # Patch sys.exit to prevent test from exiting
                github_stars_search.main()
        
        # Assertions
        mock_console_print.assert_called_once()  # Just check that console.print was called
