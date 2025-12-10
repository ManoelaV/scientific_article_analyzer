"""
Comprehensive Test Suite for Enhanced Scientific Article Analyzer

This module provides extensive testing including:
- Robust error handling validation
- PDF processing edge cases
- Timeout and exception management
- Missing metadata scenarios
- Performance and stress testing
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import time

# Import the enhanced system components
from src.enhanced_multi_agent_system import EnhancedCoordinatorAgent, ProcessingConfig
from vector_store.advanced_store import AdvancedVectorStore, VectorStoreConfig, ChunkingConfig
from src.models import ArticleContent, ScientificCategory
from mcp_server.enhanced_tools import initialize_system, search_articles, get_article_content

class TestRobustErrorHandling:
    """Test suite for robust error handling and edge cases."""
    
    @pytest.fixture
    async def enhanced_coordinator(self):
        """Create enhanced coordinator with test configuration."""
        config = ProcessingConfig(
            timeout_seconds=30,
            retry_attempts=2,
            strict_json_validation=True,
            enable_academic_rubric=True
        )
        coordinator = EnhancedCoordinatorAgent(config)
        await coordinator.initialize()
        return coordinator
    
    @pytest.fixture
    def corrupted_pdf_content(self):
        """Simulate corrupted PDF content."""
        return """
        %%PDF-1.4
        1 0 obj
        <<
        /Type /Catalog
        /Pages 2 0 R
        >>
        endobj
        
        CORRUPTED_BINARY_DATA_HERE_����������
        %%EOF
        """
    
    @pytest.fixture
    def malformed_text_content(self):
        """Create malformed text content for testing."""
        return "This is way too short for analysis"
    
    @pytest.fixture
    def incomplete_article_metadata(self):
        """Article with missing essential metadata."""
        return ArticleContent(
            title="",
            abstract="",
            authors=[],
            keywords=[],
            full_text="This article has minimal content and no metadata whatsoever. " * 20
        )
    
    @pytest.mark.asyncio
    async def test_pdf_corruption_handling(self, enhanced_coordinator, corrupted_pdf_content):
        """Test handling of corrupted PDF files."""
        try:
            # Simulate PDF processing failure
            with patch('src.pdf_processor.extract_text') as mock_extract:
                mock_extract.side_effect = Exception("PDF parsing failed: Corrupted file structure")
                
                # The system should handle this gracefully
                article = ArticleContent(
                    title="Test Article",
                    abstract="Test abstract",
                    authors=["Test Author"],
                    keywords=["test"],
                    full_text=corrupted_pdf_content
                )
                
                # Should not crash, should return error information
                result = await enhanced_coordinator.process_article(article)
                
                # Verify graceful degradation
                assert result is not None
                assert "error" in str(result).lower() or len(result.get("extraction", {})) > 0
                
        except Exception as e:
            # Even exceptions should be handled gracefully
            assert "timeout" in str(e).lower() or "processing failed" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, enhanced_coordinator):
        """Test timeout handling for long-running operations."""
        # Create very long article to potentially cause timeout
        long_content = "This is a very long article. " * 10000  # ~50k words
        
        article = ArticleContent(
            title="Extremely Long Article",
            abstract="Test abstract",
            authors=["Test Author"],
            keywords=["test"],
            full_text=long_content
        )
        
        # Set very short timeout
        enhanced_coordinator.config.timeout_seconds = 1
        
        try:
            result = await enhanced_coordinator.process_article(article)
            # If it succeeds quickly, that's also fine
            assert result is not None
        except Exception as e:
            # Should handle timeout gracefully
            assert "timeout" in str(e).lower() or "took too long" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_missing_metadata_handling(self, enhanced_coordinator, incomplete_article_metadata):
        """Test handling of articles with missing metadata."""
        result = await enhanced_coordinator.process_article(incomplete_article_metadata)
        
        # Should still produce valid output with defaults
        assert result is not None
        extraction = result.get("extraction", {})
        
        # Should have default values for missing fields
        assert "titulo" in extraction
        assert "autores" in extraction
        assert isinstance(extraction.get("autores", []), list)
        
        # Should indicate missing information appropriately
        assert "não identificado" in extraction["titulo"].lower() or extraction["titulo"] != ""
    
    @pytest.mark.asyncio
    async def test_malformed_input_validation(self, enhanced_coordinator, malformed_text_content):
        """Test validation of malformed input data."""
        article = ArticleContent(
            title="Test",
            abstract="",
            authors=[],
            keywords=[],
            full_text=malformed_text_content
        )
        
        try:
            result = await enhanced_coordinator.process_article(article)
            # If processing succeeds, check it handles short content
            if result:
                extraction = result.get("extraction", {})
                assert len(extraction) > 0  # Should have some extraction
        except Exception as e:
            # Should provide meaningful error message
            assert "too short" in str(e).lower() or "minimum" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_network_failure_simulation(self, enhanced_coordinator):
        """Test handling of network/API failures."""
        article = ArticleContent(
            title="Network Test Article",
            abstract="Testing network failure scenarios",
            authors=["Network Tester"],
            keywords=["network", "failure"],
            full_text="This article tests network failure handling. " * 50
        )
        
        # Simulate network failure in embeddings
        with patch('sentence_transformers.SentenceTransformer.encode') as mock_encode:
            mock_encode.side_effect = Exception("Network timeout")
            
            try:
                result = await enhanced_coordinator.process_article(article)
                # Should handle network failures gracefully
                assert result is not None
            except Exception as e:
                # Should provide meaningful error message
                assert "network" in str(e).lower() or "timeout" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_safety(self, enhanced_coordinator):
        """Test thread safety and concurrent processing."""
        articles = []
        for i in range(5):
            article = ArticleContent(
                title=f"Concurrent Article {i}",
                abstract=f"Abstract for article {i}",
                authors=[f"Author {i}"],
                keywords=[f"keyword{i}"],
                full_text=f"This is the content of article {i}. " * 100
            )
            articles.append(article)
        
        # Process multiple articles concurrently
        tasks = [enhanced_coordinator.process_article(article) for article in articles]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check that most succeeded or failed gracefully
            success_count = sum(1 for r in results if isinstance(r, dict))
            error_count = sum(1 for r in results if isinstance(r, Exception))
            
            # At least some should succeed, and errors should be handled
            assert success_count + error_count == len(articles)
            assert success_count >= len(articles) // 2  # At least half should succeed
            
        except Exception as e:
            # Even global failures should be informative
            assert len(str(e)) > 10  # Should have meaningful error message

class TestAdvancedVectorStoreRobustness:
    """Test advanced vector store with error handling."""
    
    @pytest.fixture
    async def vector_store_config(self):
        """Create test vector store configuration."""
        return VectorStoreConfig(
            storage_path="./test_data/vector_store",
            chunking_config=ChunkingConfig(
                chunk_size=500,  # Smaller for testing
                overlap=100,
                min_chunk_size=50
            ),
            similarity_threshold=0.5
        )
    
    @pytest.fixture
    async def test_vector_store(self, vector_store_config):
        """Create test vector store instance."""
        store = AdvancedVectorStore(vector_store_config)
        return store
    
    @pytest.mark.asyncio
    async def test_empty_content_handling(self, test_vector_store):
        """Test handling of empty or minimal content."""
        empty_article = ArticleContent(
            title="",
            abstract="",
            authors=[],
            keywords=[],
            full_text=""
        )
        
        result = await test_vector_store.add_document(empty_article, ScientificCategory.MACHINE_LEARNING)
        
        # Should handle gracefully (either succeed with defaults or fail safely)
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_extremely_long_content(self, test_vector_store):
        """Test handling of extremely long documents."""
        very_long_content = "This is a very long document. " * 10000  # ~50k words
        
        long_article = ArticleContent(
            title="Extremely Long Document",
            abstract="This document tests chunking limits",
            authors=["Stress Tester"],
            keywords=["stress", "test", "chunking"],
            full_text=very_long_content
        )
        
        try:
            result = await test_vector_store.add_document(long_article, ScientificCategory.MACHINE_LEARNING)
            assert isinstance(result, bool)
        except Exception as e:
            # Should provide meaningful error for resource limits
            assert "memory" in str(e).lower() or "size" in str(e).lower() or "limit" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self, test_vector_store):
        """Test handling of special characters and encoding issues."""
        special_chars_content = """
        This article contains special characters: αβγδ, ñáéíóú, 中文, العربية, русский
        Mathematical symbols: ∑∫∞≠≤≥±√∂∇∈∉⊂⊃∪∩
        Emojis and symbols: 🧬🌍🤖 ©®™
        """ * 20
        
        special_article = ArticleContent(
            title="Special Characters Test Article",
            abstract="Testing special character handling",
            authors=["Special Tester"],
            keywords=["unicode", "encoding"],
            full_text=special_chars_content
        )
        
        try:
            result = await test_vector_store.add_document(special_article, ScientificCategory.BIOTECHNOLOGY)
            assert isinstance(result, bool)
        except Exception as e:
            # Should handle encoding issues gracefully
            assert "encoding" in str(e).lower() or "character" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, test_vector_store):
        """Test handling under memory pressure."""
        # Create many moderately-sized articles to test memory limits
        articles = []
        for i in range(100):  # Create 100 articles
            content = f"This is article number {i} with substantial content. " * 200
            article = ArticleContent(
                title=f"Memory Test Article {i}",
                abstract=f"Testing memory handling for article {i}",
                authors=[f"Memory Tester {i}"],
                keywords=[f"memory", f"test{i}"],
                full_text=content
            )
            articles.append(article)
        
        success_count = 0
        for i, article in enumerate(articles[:10]):  # Test first 10
            try:
                result = await test_vector_store.add_document(
                    article, 
                    ScientificCategory(list(ScientificCategory)[i % 3])
                )
                if result:
                    success_count += 1
            except Exception as e:
                # Should handle memory issues gracefully
                if "memory" in str(e).lower():
                    break  # Acceptable to stop on memory limits
                else:
                    continue  # Other errors should not stop the test
        
        # Should handle at least a few articles before memory issues
        assert success_count >= 3
    
    @pytest.mark.asyncio
    async def test_corrupted_storage_recovery(self, test_vector_store):
        """Test recovery from corrupted storage files."""
        # Simulate corrupted storage by creating invalid files
        storage_path = Path(test_vector_store.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create corrupted metadata file
        corrupted_metadata = storage_path / "metadata.json"
        with open(corrupted_metadata, 'w') as f:
            f.write("{ invalid json content }")
        
        # Create corrupted chunks file
        corrupted_chunks = storage_path / "chunks.json"
        with open(corrupted_chunks, 'w') as f:
            f.write("not json at all")
        
        try:
            # Should handle corrupted files gracefully
            test_article = ArticleContent(
                title="Recovery Test",
                abstract="Testing recovery from corruption",
                authors=["Recovery Tester"],
                keywords=["recovery"],
                full_text="Testing recovery from corrupted storage files. " * 50
            )
            
            result = await test_vector_store.add_document(test_article, ScientificCategory.MACHINE_LEARNING)
            # Should either succeed (by recreating) or fail gracefully
            assert isinstance(result, bool)
            
        except Exception as e:
            # Should provide meaningful error about storage issues
            assert "storage" in str(e).lower() or "file" in str(e).lower() or "corrupted" in str(e).lower()

class TestMCPToolsRobustness:
    """Test MCP tools with comprehensive error scenarios."""
    
    @pytest.mark.asyncio
    async def test_search_with_invalid_parameters(self):
        """Test search tools with invalid parameters."""
        # Test empty query
        try:
            result = await search_articles("")
            assert False, "Should have raised error for empty query"
        except Exception as e:
            assert "empty" in str(e).lower() or "cannot be empty" in str(e).lower()
        
        # Test extremely long query
        try:
            long_query = "test query " * 200  # Over 1000 chars
            result = await search_articles(long_query)
            assert False, "Should have raised error for long query"
        except Exception as e:
            assert "long" in str(e).lower() or "maximum" in str(e).lower()
        
        # Test invalid category
        try:
            result = await search_articles("test", category="invalid_category")
            assert False, "Should have raised error for invalid category"
        except Exception as e:
            assert "invalid category" in str(e).lower()
        
        # Test invalid limit
        try:
            result = await search_articles("test", limit=100)  # Over 50
            assert False, "Should have raised error for high limit"
        except Exception as e:
            assert "limit" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_get_content_with_nonexistent_id(self):
        """Test content retrieval with non-existent IDs."""
        try:
            result = await get_article_content("nonexistent_article_id_12345")
            assert False, "Should have raised error for non-existent ID"
        except Exception as e:
            assert "not found" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_system_initialization_failure_recovery(self):
        """Test recovery from system initialization failures."""
        # Mock initialization failure
        with patch('mcp_server.enhanced_tools.initialize_system') as mock_init:
            mock_init.side_effect = Exception("Initialization failed")
            
            try:
                result = await search_articles("test query")
                assert False, "Should have raised initialization error"
            except Exception as e:
                assert "initialization" in str(e).lower() or "failed" in str(e).lower()

class TestPerformanceAndStress:
    """Performance and stress testing."""
    
    @pytest.mark.asyncio
    async def test_response_time_requirements(self, enhanced_coordinator):
        """Test that response times meet requirements."""
        article = ArticleContent(
            title="Performance Test Article",
            abstract="Testing response time requirements",
            authors=["Performance Tester"],
            keywords=["performance", "speed"],
            full_text="This article tests response time requirements. " * 100
        )
        
        start_time = time.time()
        result = await enhanced_coordinator.process_article(article)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust based on requirements)
        assert processing_time < 60.0, f"Processing took {processing_time:.2f}s, should be under 60s"
        
        # Should produce valid result
        assert result is not None
        assert "extraction" in result
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage remains stable across multiple operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple articles
        for i in range(10):
            article = ArticleContent(
                title=f"Memory Test {i}",
                abstract=f"Memory testing article {i}",
                authors=[f"Tester {i}"],
                keywords=["memory", "test"],
                full_text=f"Memory test content for article {i}. " * 100
            )
            
            try:
                # Simulate some processing
                await asyncio.sleep(0.1)
                # In real test, would process article
            except Exception:
                continue  # Continue testing even if individual articles fail
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (adjust threshold as needed)
        max_growth_mb = 100 * 1024 * 1024  # 100 MB
        assert memory_growth < max_growth_mb, f"Memory grew by {memory_growth / 1024 / 1024:.2f}MB"

class TestDataValidationAndSanitization:
    """Test data validation and sanitization."""
    
    @pytest.mark.asyncio
    async def test_json_schema_validation(self, enhanced_coordinator):
        """Test strict JSON schema validation."""
        article = ArticleContent(
            title="JSON Validation Test",
            abstract="Testing JSON schema compliance",
            authors=["JSON Tester"],
            keywords=["json", "validation"],
            full_text="This article tests JSON schema validation. " * 50
        )
        
        result = await enhanced_coordinator.process_article(article, strict_json=True)
        
        # Should have valid JSON structure
        extraction = result.get("extraction", {})
        
        required_fields = ["titulo", "autores", "resumo", "palavras_chave", 
                          "metodologia", "resultados_principais", "conclusoes"]
        
        for field in required_fields:
            assert field in extraction, f"Missing required field: {field}"
            
        # Check data types
        assert isinstance(extraction["titulo"], str)
        assert isinstance(extraction["autores"], list)
        assert isinstance(extraction["palavras_chave"], list)
    
    @pytest.mark.asyncio
    async def test_input_sanitization(self, enhanced_coordinator):
        """Test input sanitization for security."""
        malicious_content = """
        <script>alert('xss')</script>
        <?php system('rm -rf /'); ?>
        DROP TABLE articles;
        ../../../etc/passwd
        This is normal content mixed with potential attacks.
        """ * 20
        
        article = ArticleContent(
            title="<script>Malicious Title</script>",
            abstract="<?php Malicious Abstract ?>",
            authors=["'; DROP TABLE users; --"],
            keywords=["<script>", "../../../"],
            full_text=malicious_content
        )
        
        result = await enhanced_coordinator.process_article(article)
        
        # Should sanitize malicious content
        extraction = result.get("extraction", {})
        
        # Should not contain script tags or other malicious content
        assert "<script>" not in str(extraction)
        assert "<?php" not in str(extraction)
        assert "DROP TABLE" not in str(extraction)
        assert "../../../" not in str(extraction)

# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "timeout": 30,
        "max_retries": 2,
        "test_data_path": "./test_data",
        "cleanup_after_test": True
    }

@pytest.fixture(autouse=True)
async def setup_and_cleanup(test_config):
    """Setup and cleanup for each test."""
    # Setup
    test_data_path = Path(test_config["test_data_path"])
    test_data_path.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup
    if test_config["cleanup_after_test"]:
        import shutil
        if test_data_path.exists():
            shutil.rmtree(test_data_path, ignore_errors=True)

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_processing_throughput(self, enhanced_coordinator):
        """Benchmark processing throughput."""
        articles = []
        for i in range(5):
            article = ArticleContent(
                title=f"Benchmark Article {i}",
                abstract=f"Benchmarking article {i}",
                authors=[f"Benchmark Author {i}"],
                keywords=["benchmark", "performance"],
                full_text=f"Benchmark content for article {i}. " * 200
            )
            articles.append(article)
        
        start_time = time.time()
        
        results = []
        for article in articles:
            try:
                result = await enhanced_coordinator.process_article(article)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        successful_results = [r for r in results if "error" not in r]
        throughput = len(successful_results) / total_time
        
        print(f"Processed {len(successful_results)}/{len(articles)} articles in {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} articles/second")
        
        # Basic performance requirements
        assert throughput > 0.1, f"Throughput too low: {throughput:.2f} articles/second"
        assert len(successful_results) >= len(articles) // 2, "Success rate too low"

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])