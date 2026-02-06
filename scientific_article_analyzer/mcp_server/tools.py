"""
MCP Tools for Scientific Article Analysis System
Implements the two required tools: search_articles and get_article_content
Uses the main analyzer system as backend for enhanced functionality
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import json
import asyncio
import logging

from src.models import ScientificCategory

if TYPE_CHECKING:
    from main import ScientificArticleAnalyzer

logger = logging.getLogger(__name__)

class MCPTools:
    """MCP tools implementing search_articles and get_article_content with analyzer backend."""
    
    def __init__(self, analyzer: 'ScientificArticleAnalyzer'):
        """Initialize MCP tools with analyzer system backend."""
        self.analyzer = analyzer
        
        # Article cache for get_article_content
        self.article_cache = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the analyzer system and vector store."""
        if not self._initialized:
            await self.analyzer.initialize()
            self._initialized = True
            logger.info("Analyzer system and vector store initialized")
    
    async def search_articles(self, query: str) -> List[Dict[str, Any]]:
        """Search for articles in the vector store.
        
        Args:
            query: Search query string
            
        Returns:
            List of articles with id, title, area, and score
        """
        try:
            # Ensure analyzer system is initialized
            await self.initialize()
            
            # Use analyzer system to search for similar articles
            results = await self.analyzer.vector_store.search_similar(
                query=query,
                category=None,  # Search all categories
                limit=10,
                min_similarity=0.1
            )
            
            # Format results as specified: [{id, title, area, score}]
            formatted_results = []
            for i, result in enumerate(results):
                article_id = result.get("id", f"article_{i}")
                title = result.get("title", result.get("metadata", {}).get("title", "Unknown Title"))
                category = result.get("category", result.get("metadata", {}).get("category", "unknown"))
                score = result.get("similarity", result.get("score", 0.0))
                content = result.get("content", result.get("text", ""))
                
                formatted_result = {
                    "id": article_id,
                    "title": title,
                    "area": category if isinstance(category, str) else category.value,
                    "score": round(float(score), 3)
                }
                formatted_results.append(formatted_result)
                
                # Cache article content for get_article_content
                self.article_cache[article_id] = {
                    "id": article_id,
                    "title": title,
                    "area": category if isinstance(category, str) else category.value,
                    "content": content
                }
            
            logger.info(f"Search for '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [{
                "error": f"Search failed: {str(e)}"
            }]
    
    async def get_article_content(self, id: str) -> Dict[str, Any]:
        """Get full content of an article by ID.
        
        Args:
            id: Article ID from search results
            
        Returns:
            Dictionary with id, title, area, and content
        """
        try:
            # Check if article is in cache first
            if id in self.article_cache:
                return self.article_cache[id]
            
            # Ensure multi-agent system is initialized
            await self.initialize()
            
            # Search through all categories to find the article using multi-agent system
            for category in ScientificCategory:
                articles = self.multi_agent_system.vector_store.get_reference_articles(category)
                for article in articles:
                    if article.get('id') == id:
                        result = {
                            "id": id,
                            "title": article.get('title', ''),
                            "area": category.value,
                            "content": article.get('content', '')
                        }
                        
                        # Cache for future use
                        self.article_cache[id] = result
                        return result
            
            # Article not found
            logger.warning(f"Article with ID '{id}' not found")
            return {
                "error": f"Article with ID '{id}' not found"
            }
            
        except Exception as e:
            logger.error(f"Failed to get article content for ID '{id}': {e}")
            return {
                "error": f"Failed to get article content: {str(e)}"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store and cached articles."""
        stats = {
            "vector_store_initialized": self._initialized,
            "cached_articles": len(self.article_cache),
            "available_categories": [cat.value for cat in ScientificCategory]
        }
        
        if self._initialized:
            stats.update(self.multi_agent_system.vector_store.get_stats())
        
        return stats