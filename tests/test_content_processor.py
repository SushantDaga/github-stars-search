"""
Tests for the content processor.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import the module to test
from src.processor.content_processor import ContentProcessor

# Sample test data
SAMPLE_REPO = {
    "id": 12345,
    "name": "test-repo",
    "full_name": "test-user/test-repo",
    "description": "A test repository"
}

SAMPLE_README_MARKDOWN = """
# Test Repository

This is a test repository for unit tests.

## Section 1

This is the first section of the README.

## Section 2

This is the second section of the README.

### Subsection 2.1

This is a subsection.

## Section 3

This is the third section of the README.
"""

SAMPLE_README_NO_HEADERS = """
This is a test repository README without any headers.
It just contains plain text content.
This is used to test the sliding window chunking strategy.
"""

SAMPLE_README_LARGE = """
# Large Test Repository

This is a large test repository README.

## Section 1

""" + "This is a very long section with repeated text. " * 100 + """

## Section 2

This is the second section.
"""

@pytest.fixture
def content_processor():
    """Create a content processor instance for testing."""
    config = {
        "max_readme_size": 10000,
        "chunk_strategy": "hybrid",
        "max_chunk_size": 100,
        "chunk_overlap": 20
    }
    return ContentProcessor(config)

@pytest.mark.unit
class TestContentProcessorInit:
    """Test the initialization of the content processor."""
    
    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = {
            "max_readme_size": 20000,
            "chunk_strategy": "semantic",
            "max_chunk_size": 200,
            "chunk_overlap": 30
        }
        processor = ContentProcessor(config)
        
        assert processor.max_readme_size == 20000
        assert processor.chunk_strategy == "semantic"
        assert processor.max_chunk_size == 200
        assert processor.chunk_overlap == 30
    
    def test_init_with_empty_config(self):
        """Test initialization with empty configuration."""
        processor = ContentProcessor({})
        
        assert processor.max_readme_size == 500000  # Default value
        assert processor.chunk_strategy == "hybrid"  # Default value
        assert processor.max_chunk_size == 512  # Default value
        assert processor.chunk_overlap == 50  # Default value

@pytest.mark.unit
class TestCleanContent:
    """Test the _clean_content method."""
    
    def test_clean_content(self, content_processor):
        """Test cleaning and normalizing content."""
        # Call the method
        cleaned = content_processor._clean_content(SAMPLE_README_MARKDOWN)
        
        # Assertions
        assert "Test Repository" in cleaned
        assert "Section 1" in cleaned
        assert "Section 2" in cleaned
        assert "Section 3" in cleaned
        
        # Check that excessive newlines are removed
        assert "\n\n\n" not in cleaned
    
    def test_clean_content_with_html(self, content_processor):
        """Test cleaning content with HTML."""
        html_content = """
        <h1>Test Repository</h1>
        <p>This is a test repository with <strong>HTML</strong> content.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        """
        
        # Call the method
        cleaned = content_processor._clean_content(html_content)
        
        # Assertions
        assert "Test Repository" in cleaned
        assert "HTML" in cleaned
        assert "Item 1" in cleaned
        assert "Item 2" in cleaned

@pytest.mark.unit
class TestSemanticChunking:
    """Test the _semantic_chunking method."""
    
    def test_semantic_chunking(self, content_processor):
        """Test chunking content based on semantic sections."""
        # Call the method
        chunks = content_processor._semantic_chunking(SAMPLE_README_MARKDOWN, SAMPLE_REPO)
        
        # Assertions
        assert len(chunks) >= 3  # At least one chunk per main section
        
        # Check that each chunk has the expected metadata
        for chunk in chunks:
            assert chunk["id"].startswith(f"{SAMPLE_REPO['id']}-")
            assert chunk["repo_id"] == SAMPLE_REPO["id"]
            assert chunk["repo_name"] == SAMPLE_REPO["full_name"]
            assert "content" in chunk
            assert chunk["chunk_type"] == "readme_section"
            assert "section_index" in chunk
            
            # Check that repository context is included
            assert f"Repository: {SAMPLE_REPO['full_name']}" in chunk["content"]
    
    def test_semantic_chunking_no_headers(self, content_processor):
        """Test semantic chunking with content that has no headers."""
        # Call the method
        chunks = content_processor._semantic_chunking(SAMPLE_README_NO_HEADERS, SAMPLE_REPO)
        
        # Should fall back to sliding window
        assert len(chunks) > 0
        assert chunks[0]["chunk_type"] == "readme_window"

@pytest.mark.unit
class TestSlidingWindowChunking:
    """Test the _sliding_window_chunking method."""
    
    def test_sliding_window_chunking(self, content_processor):
        """Test chunking content using sliding window approach."""
        # Call the method
        chunks = content_processor._sliding_window_chunking(SAMPLE_README_NO_HEADERS, SAMPLE_REPO)
        
        # Assertions
        assert len(chunks) > 0
        
        # Check that each chunk has the expected metadata
        for chunk in chunks:
            assert chunk["id"].startswith(f"{SAMPLE_REPO['id']}-")
            assert chunk["repo_id"] == SAMPLE_REPO["id"]
            assert chunk["repo_name"] == SAMPLE_REPO["full_name"]
            assert "content" in chunk
            assert chunk["chunk_type"] == "readme_window"
            assert "window_index" in chunk
            
            # Check that repository context is included
            assert f"Repository: {SAMPLE_REPO['full_name']}" in chunk["content"]
    
    def test_sliding_window_chunking_overlap(self, content_processor):
        """Test that sliding window chunks have the expected overlap."""
        # Create a processor with specific chunk size and overlap
        config = {
            "max_chunk_size": 10,  # Small size for testing
            "chunk_overlap": 5     # 50% overlap
        }
        processor = ContentProcessor(config)
        
        # Simple content with distinct words
        content = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15"
        
        # Call the method
        chunks = processor._sliding_window_chunking(content, SAMPLE_REPO)
        
        # Assertions
        assert len(chunks) > 1
        
        # Extract just the content without the repository context
        chunk_contents = [chunk["content"].split("\n\n")[1] for chunk in chunks]
        
        # Check that the second chunk starts with words from the first chunk
        words_in_first_chunk = chunk_contents[0].split()
        words_in_second_chunk = chunk_contents[1].split()
        
        # The first N words of the second chunk should be the last N words of the first chunk
        # where N is the overlap size
        assert words_in_second_chunk[:5] == words_in_first_chunk[-5:]

@pytest.mark.unit
class TestHybridChunking:
    """Test the _hybrid_chunking method."""
    
    def test_hybrid_chunking(self, content_processor):
        """Test chunking content using hybrid approach."""
        # Call the method
        chunks = content_processor._hybrid_chunking(SAMPLE_README_MARKDOWN, SAMPLE_REPO)
        
        # Assertions
        assert len(chunks) > 0
        
        # Check that each chunk has the expected metadata
        for chunk in chunks:
            assert chunk["id"].startswith(f"{SAMPLE_REPO['id']}-")
            assert chunk["repo_id"] == SAMPLE_REPO["id"]
            assert chunk["repo_name"] == SAMPLE_REPO["full_name"]
            assert "content" in chunk
            assert chunk["chunk_type"] in ["readme_section", "readme_hybrid"]
            
            # Check that repository context is included
            assert f"Repository: {SAMPLE_REPO['full_name']}" in chunk["content"]
    
    def test_hybrid_chunking_large_section(self, content_processor):
        """Test hybrid chunking with a large section that needs to be split."""
        # Call the method
        chunks = content_processor._hybrid_chunking(SAMPLE_README_LARGE, SAMPLE_REPO)
        
        # Assertions
        assert len(chunks) > 2  # Should have more chunks than sections due to splitting
        
        # Check for hybrid chunks
        hybrid_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "readme_hybrid"]
        assert len(hybrid_chunks) > 0
        
        # Check that hybrid chunks have both section_index and window_index
        for chunk in hybrid_chunks:
            assert "section_index" in chunk
            assert "window_index" in chunk

@pytest.mark.unit
class TestProcessReadme:
    """Test the process_readme method."""
    
    def test_process_readme_semantic(self):
        """Test processing README with semantic chunking."""
        config = {
            "chunk_strategy": "semantic",
            "max_chunk_size": 100,
            "chunk_overlap": 20
        }
        processor = ContentProcessor(config)
        
        # Call the method
        chunks = processor.process_readme(SAMPLE_README_MARKDOWN, SAMPLE_REPO)
        
        # Assertions
        assert len(chunks) > 0
        assert all(chunk["chunk_type"] == "readme_section" for chunk in chunks)
    
    def test_process_readme_sliding(self):
        """Test processing README with sliding window chunking."""
        config = {
            "chunk_strategy": "sliding",
            "max_chunk_size": 100,
            "chunk_overlap": 20
        }
        processor = ContentProcessor(config)
        
        # Call the method
        chunks = processor.process_readme(SAMPLE_README_MARKDOWN, SAMPLE_REPO)
        
        # Assertions
        assert len(chunks) > 0
        assert all(chunk["chunk_type"] == "readme_window" for chunk in chunks)
    
    def test_process_readme_hybrid(self):
        """Test processing README with hybrid chunking."""
        config = {
            "chunk_strategy": "hybrid",
            "max_chunk_size": 100,
            "chunk_overlap": 20
        }
        processor = ContentProcessor(config)
        
        # Call the method
        chunks = processor.process_readme(SAMPLE_README_MARKDOWN, SAMPLE_REPO)
        
        # Assertions
        assert len(chunks) > 0
    
    def test_process_readme_truncation(self):
        """Test that large READMEs are truncated."""
        config = {
            "max_readme_size": 100,  # Very small for testing
            "chunk_strategy": "hybrid"
        }
        processor = ContentProcessor(config)
        
        # Call the method with a large README
        chunks = processor.process_readme(SAMPLE_README_LARGE, SAMPLE_REPO)
        
        # Assertions
        assert len(chunks) > 0
        
        # Check that the content was truncated
        total_content = "".join(chunk["content"] for chunk in chunks)
        assert len(total_content) < len(SAMPLE_README_LARGE) + len(chunks) * len(f"Repository: {SAMPLE_REPO['full_name']}\n\n")

@pytest.mark.unit
class TestProcessDescription:
    """Test the process_description method."""
    
    def test_process_description(self, content_processor):
        """Test processing repository description."""
        # Call the method
        chunk = content_processor.process_description(SAMPLE_REPO)
        
        # Assertions
        assert chunk is not None
        assert chunk["id"] == f"{SAMPLE_REPO['id']}-description"
        assert chunk["repo_id"] == SAMPLE_REPO["id"]
        assert chunk["repo_name"] == SAMPLE_REPO["full_name"]
        assert chunk["chunk_type"] == "description"
        assert f"Repository: {SAMPLE_REPO['full_name']}" in chunk["content"]
        assert f"Description: {SAMPLE_REPO['description']}" in chunk["content"]
    
    def test_process_description_no_description(self, content_processor):
        """Test processing repository with no description."""
        repo_no_desc = {
            "id": 12345,
            "name": "test-repo",
            "full_name": "test-user/test-repo"
        }
        
        # Call the method
        chunk = content_processor.process_description(repo_no_desc)
        
        # Assertions
        assert chunk is None
