"""
GitHub API client for fetching starred repositories and their content.
"""

import os
import time
import base64
import logging
import requests
from typing import List, Dict, Any, Optional
from github import Github, GithubException
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

class GitHubClient:
    """
    Client for interacting with the GitHub API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GitHub client.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.api_key = os.environ.get("GITHUB_STARS_KEY")
        self.per_page = config.get("per_page", 100)
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 30)
        
        # Initialize GitHub client
        self.github = Github(self.api_key, per_page=self.per_page, timeout=self.timeout)
        
        # Initialize session for REST API requests
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {self.api_key}",
            "Accept": "application/vnd.github.v3+json"
        })
    
    def get_starred_repositories(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the user's starred repositories.
        
        Args:
            limit (int, optional): Maximum number of repositories to fetch
        
        Returns:
            list: List of repository dictionaries
        """
        logger.info("Fetching starred repositories")
        
        repositories = []
        user = self.github.get_user()
        
        try:
            # Get starred repositories
            starred = user.get_starred()
            
            # Determine how many to fetch
            total_count = starred.totalCount
            if limit is not None:
                total_count = min(total_count, limit)
            
            logger.info(f"Found {total_count} starred repositories")
            
            # Fetch repositories with pagination
            for i, repo in enumerate(starred):
                if limit is not None and i >= limit:
                    break
                
                # Extract repository data
                repo_data = {
                    "id": repo.id,
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "html_url": repo.html_url,
                    "clone_url": repo.clone_url,
                    "language": repo.language,
                    "stargazers_count": repo.stargazers_count,
                    "watchers_count": repo.watchers_count,
                    "forks_count": repo.forks_count,
                    "open_issues_count": repo.open_issues_count,
                    "topics": repo.topics,
                    "created_at": repo.created_at.isoformat() if repo.created_at else None,
                    "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                    "pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
                    "size": repo.size,
                    "default_branch": repo.default_branch,
                    "license": repo.license.key if repo.license else None,
                    "owner": {
                        "login": repo.owner.login,
                        "id": repo.owner.id,
                        "avatar_url": repo.owner.avatar_url,
                        "html_url": repo.owner.html_url
                    }
                }
                
                repositories.append(repo_data)
                
                # Avoid rate limiting
                if (i + 1) % 100 == 0:
                    logger.info(f"Fetched {i + 1} repositories")
                    time.sleep(1)
            
            return repositories
        
        except GithubException as e:
            logger.error(f"GitHub API error: {str(e)}")
            raise
    
    def get_readme(self, repo_full_name: str) -> Optional[str]:
        """
        Get the README content for a repository.
        
        Args:
            repo_full_name (str): Full name of the repository (owner/repo)
        
        Returns:
            str: README content or None if not found
        """
        logger.info(f"Fetching README for {repo_full_name}")
        
        # Try to get the default README first
        readme_content = self._get_readme_content(repo_full_name)
        if readme_content:
            # Try to detect language
            try:
                lang = detect(readme_content)
                if lang == "en":
                    return readme_content
                logger.info(f"README for {repo_full_name} is not in English (detected: {lang})")
            except LangDetectException:
                # If language detection fails, assume it's usable
                return readme_content
        
        # If no README or not in English, try to find an English README
        english_readme = self._find_english_readme(repo_full_name)
        if english_readme:
            return english_readme
        
        # If no English README found, return the default README
        return readme_content
    
    def _get_readme_content(self, repo_full_name: str) -> Optional[str]:
        """
        Get the content of the default README file.
        
        Args:
            repo_full_name (str): Full name of the repository (owner/repo)
        
        Returns:
            str: README content or None if not found
        """
        for attempt in range(self.max_retries):
            try:
                # Try to get the repository README
                repo = self.github.get_repo(repo_full_name)
                readme = repo.get_readme()
                content = base64.b64decode(readme.content).decode("utf-8")
                return content
            
            except GithubException as e:
                if e.status == 404:
                    logger.warning(f"No README found for {repo_full_name}")
                    return None
                elif attempt < self.max_retries - 1:
                    logger.warning(f"GitHub API error, retrying: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"GitHub API error: {str(e)}")
                    return None
    
    def _find_english_readme(self, repo_full_name: str) -> Optional[str]:
        """
        Try to find an English README file in the repository.
        
        Args:
            repo_full_name (str): Full name of the repository (owner/repo)
        
        Returns:
            str: README content or None if not found
        """
        # List of possible English README filenames
        english_readme_names = [
            "README.en.md",
            "README_EN.md",
            "README-en.md",
            "README-EN.md",
            "README_ENGLISH.md",
            "README-ENGLISH.md",
            "README.english.md",
            "README-english.md",
            "en/README.md",
            "english/README.md"
        ]
        
        # Try each possible filename
        for readme_name in english_readme_names:
            url = f"https://api.github.com/repos/{repo_full_name}/contents/{readme_name}"
            
            for attempt in range(self.max_retries):
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    
                    if response.status_code == 200:
                        content_data = response.json()
                        if content_data.get("encoding") == "base64" and content_data.get("content"):
                            content = base64.b64decode(content_data["content"]).decode("utf-8")
                            logger.info(f"Found English README ({readme_name}) for {repo_full_name}")
                            return content
                    
                    # If not found, break the retry loop for this filename
                    break
                
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Error fetching {readme_name}, retrying: {str(e)}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"Error fetching {readme_name}: {str(e)}")
        
        # No English README found
        return None
