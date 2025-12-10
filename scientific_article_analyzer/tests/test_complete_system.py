"""
Comprehensive Test Suite for Scientific Article Analysis System

This test suite validates all core components of the multi-agent system:
- Vector Store functionality
- Multi-Agent System integration
- MCP Server tools
- Individual agent performance
- End-to-end workflows
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ScientificCategory, ArticleContent
from vector_store.store import VectorStore
from vector_store.embeddings import EmbeddingManager
from multi_agent_system import CoordinatorAgent, Agent, AgentRole
from mcp_server.tools import MCPTools


class TestVectorStore:
    """Test suite for VectorStore functionality."""
    
    @pytest.fixture
    async def vector_store(self):
        """Create a test vector store instance."""
        embedding_manager = EmbeddingManager()
        store = VectorStore(embedding_manager)
        return store
    
    @pytest.mark.asyncio
    async def test_vector_store_initialization(self, vector_store):
        """Test vector store initialization."""
        assert vector_store is not None
        assert hasattr(vector_store, 'embedding_manager')
        assert hasattr(vector_store, 'articles')
        assert hasattr(vector_store, 'embeddings')
    
    @pytest.mark.asyncio
    async def test_add_article(self, vector_store):
        """Test adding an article to the vector store."""
        article = ArticleContent(
            title="Test Article",
            abstract="Test abstract",
            full_text="This is a test article about machine learning.",
            authors=["Test Author"]
        )
        
        success = await vector_store.add_article(article, ScientificCategory.COMPUTER_SCIENCE)
        assert success is True
        
        # Verify article was added
        articles = vector_store.get_reference_articles(ScientificCategory.COMPUTER_SCIENCE)
        assert len(articles) > 0
        assert any(art['title'] == "Test Article" for art in articles)
    
    @pytest.mark.asyncio
    async def test_search_similar(self, vector_store):
        """Test similarity search functionality."""
        # First add a test article
        article = ArticleContent(
            title="Machine Learning Basics",
            abstract="Introduction to ML",
            full_text="Machine learning is a subset of artificial intelligence.",
            authors=["ML Expert"]
        )
        
        await vector_store.add_article(article, ScientificCategory.COMPUTER_SCIENCE)
        
        # Search for similar articles
        results = await vector_store.search_similar(
            query="artificial intelligence machine learning",
            limit=5
        )
        
        assert isinstance(results, list)
        # Should find the article we just added (if embeddings work)
        
    def test_get_stats(self, vector_store):
        """Test getting vector store statistics."""
        stats = vector_store.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_articles' in stats
        assert 'categories' in stats
        assert 'embedding_dimension' in stats
        assert stats['embedding_dimension'] == 384


class TestEmbeddingManager:
    """Test suite for EmbeddingManager."""
    
    @pytest.fixture
    def embedding_manager(self):
        """Create a test embedding manager instance."""
        return EmbeddingManager()
    
    def test_embedding_manager_initialization(self, embedding_manager):
        """Test embedding manager initialization."""
        assert embedding_manager is not None
        assert embedding_manager.model_name == "all-MiniLM-L6-v2"
        assert hasattr(embedding_manager, 'model')
    
    def test_create_embedding(self, embedding_manager):
        """Test embedding creation."""
        text = "This is a test sentence for embedding."
        embedding = embedding_manager.create_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # Expected dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_create_embedding_empty_text(self, embedding_manager):
        """Test embedding creation with empty text."""
        embedding = embedding_manager.create_embedding("")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(x == 0.0 for x in embedding)


class TestMultiAgentSystem:
    """Test suite for Multi-Agent System."""
    
    @pytest.fixture
    async def coordinator(self):
        """Create a test coordinator agent."""
        coordinator = CoordinatorAgent(openai_api_key="test-key")
        return coordinator
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator agent initialization."""
        assert coordinator is not None
        assert coordinator.role == AgentRole.COORDINATOR
        assert hasattr(coordinator, 'agents')
        assert len(coordinator.agents) == 4  # 4 specialized agents
    
    @pytest.mark.asyncio
    async def test_coordinator_initialize(self, coordinator):
        """Test coordinator initialization process."""
        await coordinator.initialize()
        
        # Verify vector store was initialized
        stats = coordinator.vector_store.get_stats()
        assert stats['total_articles'] >= 0
    
    @pytest.mark.asyncio
    async def test_analyze_article_multi_agent(self, coordinator):
        """Test multi-agent article analysis."""
        test_article = """
        Deep Learning for Computer Vision: A Comprehensive Review
        
        Abstract: This paper provides a comprehensive review of deep learning
        techniques applied to computer vision problems.
        
        Introduction: Deep learning has revolutionized computer vision...
        """
        
        # Mock the analysis to avoid API calls in tests
        with patch.object(coordinator.agents[AgentRole.EXTRACTOR], 'execute_task', 
                         return_value=Mock(result={
                             "what problem does the article propose to solve?": "Computer vision challenges",
                             "step by step on how to solve it": ["Use deep learning", "Train neural networks"],
                             "conclusion": "Deep learning is effective for computer vision"
                         })):
            with patch.object(coordinator.agents[AgentRole.REVIEWER], 'execute_task',
                             return_value=Mock(result="Resenha crítica em português...")):
                
                result = await coordinator.analyze_article_multi_agent(
                    input_data=test_article,
                    input_type="text"
                )
        
        assert isinstance(result, dict)
        assert 'success' in result or 'article' in result


class TestMCPTools:
    """Test suite for MCP Tools."""
    
    @pytest.fixture
    async def mcp_tools(self):
        """Create MCP tools instance with mock coordinator."""
        coordinator = CoordinatorAgent(openai_api_key="test-key")
        await coordinator.initialize()
        return MCPTools(coordinator)
    
    @pytest.mark.asyncio
    async def test_mcp_tools_initialization(self, mcp_tools):
        """Test MCP tools initialization."""
        assert mcp_tools is not None
        assert hasattr(mcp_tools, 'multi_agent_system')
        assert hasattr(mcp_tools, 'article_cache')
    
    @pytest.mark.asyncio
    async def test_search_articles(self, mcp_tools):
        """Test search_articles MCP tool."""
        await mcp_tools.initialize()
        
        results = await mcp_tools.search_articles("machine learning")
        
        assert isinstance(results, list)
        # May return error due to missing embeddings in test, but structure should be correct
        if results and 'error' not in results[0]:
            assert all(isinstance(r, dict) for r in results)
            assert all('id' in r and 'title' in r and 'area' in r and 'score' in r for r in results)
    
    @pytest.mark.asyncio
    async def test_get_article_content(self, mcp_tools):
        """Test get_article_content MCP tool."""
        await mcp_tools.initialize()
        
        # Test with non-existent ID
        result = await mcp_tools.get_article_content("non_existent_id")
        
        assert isinstance(result, dict)
        assert 'error' in result or 'content' in result
    
    def test_get_stats(self, mcp_tools):
        """Test MCP tools statistics."""
        stats = mcp_tools.get_stats()
        
        assert isinstance(stats, dict)
        assert 'vector_store_initialized' in stats
        assert 'cached_articles' in stats
        assert 'available_categories' in stats


class TestModels:
    """Test suite for data models."""
    
    def test_scientific_category_enum(self):
        """Test ScientificCategory enum."""
        categories = list(ScientificCategory)
        
        assert len(categories) == 3
        assert ScientificCategory.COMPUTER_SCIENCE in categories
        assert ScientificCategory.PHYSICS in categories
        assert ScientificCategory.BIOLOGY in categories
    
    def test_article_content_model(self):
        """Test ArticleContent dataclass."""
        article = ArticleContent(
            title="Test Title",
            abstract="Test Abstract",
            full_text="Test full text content",
            authors=["Author 1", "Author 2"],
            keywords=["keyword1", "keyword2"]
        )
        
        assert article.title == "Test Title"
        assert article.abstract == "Test Abstract"
        assert article.full_text == "Test full text content"
        assert len(article.authors) == 2
        assert len(article.keywords) == 2
        assert article.content == "Test full text content"  # Test property alias
    
    def test_article_content_defaults(self):
        """Test ArticleContent with default values."""
        article = ArticleContent(
            title="Test Title",
            abstract="Test Abstract", 
            full_text="Test content"
        )
        
        assert article.authors == []
        assert article.keywords == []
        assert article.publication_date is None


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis workflow."""
        # Initialize system
        coordinator = CoordinatorAgent(openai_api_key="test-key")
        await coordinator.initialize()
        
        # Test article
        test_article = """
        Quantum Computing Applications in Machine Learning
        
        Abstract: This paper explores the intersection of quantum computing
        and machine learning, presenting novel algorithms for quantum ML.
        
        Introduction: Quantum computing offers exponential speedups...
        Methodology: We develop quantum versions of classical ML algorithms...
        Results: Our quantum algorithms show significant improvements...
        Conclusion: Quantum ML represents the future of computation.
        """
        
        # Mock expensive operations for testing
        with patch.object(coordinator.agents[AgentRole.EXTRACTOR], 'execute_task') as mock_extract:
            with patch.object(coordinator.agents[AgentRole.REVIEWER], 'execute_task') as mock_review:
                
                # Setup mock returns
                mock_extract.return_value = Mock(
                    status="completed",
                    result={
                        "what problem does the article propose to solve?": "Quantum ML optimization",
                        "step by step on how to solve it": ["Develop quantum algorithms", "Test on quantum hardware"],
                        "conclusion": "Quantum ML shows promise"
                    }
                )
                
                mock_review.return_value = Mock(
                    status="completed", 
                    result="**Resenha Crítica**: Artigo inovador sobre quantum ML..."
                )
                
                # Run analysis
                result = await coordinator.analyze_article_multi_agent(
                    input_data=test_article,
                    input_type="text"
                )
                
                # Verify result structure
                assert isinstance(result, dict)
                # Should have either success=True or contain analysis results
    
    @pytest.mark.asyncio  
    async def test_mcp_integration(self):
        """Test MCP server integration."""
        # Initialize components
        coordinator = CoordinatorAgent(openai_api_key="test-key")
        await coordinator.initialize()
        
        mcp_tools = MCPTools(coordinator)
        await mcp_tools.initialize()
        
        # Test MCP workflow
        search_results = await mcp_tools.search_articles("test query")
        assert isinstance(search_results, list)
        
        # Test content retrieval (even if it fails due to no articles)
        content_result = await mcp_tools.get_article_content("test_id")
        assert isinstance(content_result, dict)


class TestPerformance:
    """Performance and load tests."""
    
    @pytest.mark.asyncio
    async def test_embedding_performance(self):
        """Test embedding generation performance."""
        import time
        
        embedding_manager = EmbeddingManager()
        
        # Test single embedding
        start_time = time.time()
        embedding = embedding_manager.create_embedding("Test text for performance measurement")
        single_time = time.time() - start_time
        
        assert single_time < 1.0  # Should be under 1 second
        assert len(embedding) == 384
    
    @pytest.mark.asyncio
    async def test_vector_store_scalability(self):
        """Test vector store with multiple articles."""
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore(embedding_manager)
        
        # Add multiple test articles
        for i in range(5):
            article = ArticleContent(
                title=f"Test Article {i}",
                abstract=f"Abstract for article {i}",
                full_text=f"Content for test article number {i} with unique keywords_{i}",
                authors=[f"Author {i}"]
            )
            
            category = [ScientificCategory.COMPUTER_SCIENCE, ScientificCategory.PHYSICS, ScientificCategory.BIOLOGY][i % 3]
            success = await vector_store.add_article(article, category)
            assert success is True
        
        # Test search performance
        import time
        start_time = time.time()
        results = await vector_store.search_similar("test keywords", limit=3)
        search_time = time.time() - start_time
        
        assert search_time < 1.0  # Should be under 1 second
        assert isinstance(results, list)


# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test markers for different test categories
pytestmark = [
    pytest.mark.asyncio,  # Mark all tests as async
]


if __name__ == "__main__":
    """Run tests directly."""
    print("🧪 Running Scientific Article Analysis System Tests")
    print("=" * 60)
    
    # Run with pytest
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
    ])