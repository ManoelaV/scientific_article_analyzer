#!/usr/bin/env python3
"""
Scientific Article Analyzer MCP Server

A Model Context Protocol server that provides tools for analyzing scientific articles:
- Article classification into scientific categories
- Information extraction in structured JSON format  
- Critical review generation
- Vector store search for similar articles
- Reference article management
"""

import asyncio
import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from mcp_server.tools import MCPTools
from multi_agent_system import CoordinatorAgent
from vector_store import VectorStore

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store_db")
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))


def create_server() -> FastMCP:
    """Create and configure the MCP server."""
    
    api_key = OPENAI_API_KEY
    if not api_key:
        print("Warning: No OPENAI_API_KEY provided. Some features may not work correctly.")
        api_key = "test-key"  # Use test key for development
    
    # Create FastMCP server
    app = FastMCP(
        name="scientific-article-analyzer",
        instructions="Scientific Article Analyzer MCP Server - Provides AI-powered analysis of scientific articles including classification, information extraction, and critical review generation using a multi-agent system."
    )
    
    # Initialize multi-agent system
    multi_agent_system = CoordinatorAgent(
        openai_api_key=api_key
    )
    
    # Initialize MCP tools with multi-agent backend
    tools = MCPTools(multi_agent_system)
    
    # Register search_articles tool
    @app.tool()
    async def search_articles(query: str) -> list:
        """Search for articles in the vector store.
        
        Args:
            query: Search query string
            
        Returns:
            List of articles with id, title, area, and score
        """
        return await tools.search_articles(query)
    
    # Register get_article_content tool
    @app.tool()
    async def get_article_content(id: str) -> dict:
        """Get full content of an article by ID.
        
        Args:
            id: Article ID from search results
            
        Returns:
            Dictionary with id, title, area, and content
        """
        return await tools.get_article_content(id)
    
    # Register multi-agent analysis tool
    @app.tool()
    async def analyze_article(
        content: str, 
        input_type: str = "text",
        generate_review: bool = True
    ) -> dict:
        """Complete article analysis using multi-agent system.
        
        Args:
            content: Article content (text, PDF path, or URL)
            input_type: Type of input ("text", "pdf", or "url")
            generate_review: Whether to generate critical review
            
        Returns:
            Complete analysis including classification, extraction, and review
        """
        return await multi_agent_system.analyze_article_multi_agent(
            input_data=content,
            input_type=input_type
        )
    
    return app


async def initialize_vector_store():
    """Initialize the vector store with sample reference articles if empty."""
    try:
        vector_store = VectorStore(VECTOR_STORE_PATH)
        
        # Check if vector store has reference articles
        stats = await vector_store.get_article_count()
        total_references = sum(cat_data.get("reference_articles", 0) for cat_data in stats.values())
        
        if total_references < 9:  # Need 3 articles per category
            print("Initializing vector store with sample reference articles...")
            await vector_store.initialize_with_sample_articles()
            print("Vector store initialized successfully!")
        else:
            print(f"Vector store already contains {total_references} reference articles.")
            
    except Exception as e:
        print(f"Warning: Could not initialize vector store: {e}")
        print("The server will still start, but classification may be less accurate.")


def main():
    """Main entry point for the MCP server."""
    try:
        # Create the server
        app = create_server()
        
        # Initialize vector store in the background
        asyncio.create_task(initialize_vector_store())
        
        print(f"Scientific Article Analyzer MCP Server")
        print(f"Vector Store Path: {VECTOR_STORE_PATH}")
        print(f"Available Tools:")
        print(f"  - search_articles: Search for articles in vector store")
        print(f"  - get_article_content: Get full content by article ID")
        print(f"  - analyze_article: Complete multi-agent analysis")
        print(f"")
        print(f"Multi-Agent System:")
        print(f"  - CoordinatorAgent: Orchestrates analysis workflow")
        print(f"  - ArticleProcessorAgent: Handles PDF/URL processing")
        print(f"  - ClassifierAgent: Scientific category classification")
        print(f"  - ExtractorAgent: Structured information extraction")
        print(f"  - ReviewerAgent: Critical review generation (Portuguese)")
        print(f"")
        print(f"Supported Scientific Categories:")
        print(f"  - Computer Science")
        print(f"  - Physics")
        print(f"  - Biology")
        print(f"")
        print(f"Supported Input Types:")
        print(f"  - PDF files (local path)")
        print(f"  - URLs (arXiv, PubMed, etc.)")
        print(f"  - Raw text")
        print(f"")
        
        # Run the server
        app.run()
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()