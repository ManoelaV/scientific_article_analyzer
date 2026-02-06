#!/usr/bin/env python3
"""
Scientific Article Analysis System - Main Application
Integrates all components for complete article analysis workflow
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

# Add the project directory to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models import ScientificCategory, AnalysisResult, VectorStoreEntry
from src.article_processor import ArticleProcessor
from src.classifier import ArticleClassifier
from src.extractor import InformationExtractor
from src.reviewer import CriticalReviewer
from vector_store.store import VectorStore
from vector_store.embeddings import EmbeddingManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScientificArticleAnalyzer:
    """
    Main system class that orchestrates the complete article analysis workflow.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None, api_base: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the complete analysis system.
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
            anthropic_api_key: Anthropic API key (alternative to OpenAI)
            api_base: Base URL for custom API endpoints (e.g., Ollama, LM Studio)
            model: Model name to use (default: gpt-4o-mini)
        """
        self.processor = ArticleProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(self.embedding_manager)
        
        # Initialize classification and analysis components (with API keys)
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.api_base = api_base
        self.model = model
        
        # Components will be initialized when API keys are available
        self.classifier = None
        self.extractor = None 
        self.reviewer = None
        
        self.initialized = False

    async def initialize(self):
        """
        Initialize the system with reference articles for classification.
        """
        if self.initialized:
            return
        
        logger.info("Initializing Scientific Article Analysis System...")
        
        try:
            # Initialize vector store with sample articles
            await self.vector_store.initialize_with_sample_articles()
            
            # Initialize AI components if API keys are available
            if self.openai_api_key:
                from src.classifier import ArticleClassifier
                from src.extractor import InformationExtractor
                from src.reviewer import CriticalReviewer
                
                self.classifier = ArticleClassifier(openai_api_key=self.openai_api_key, api_base=self.api_base, model=self.model)
                self.extractor = InformationExtractor(openai_api_key=self.openai_api_key, api_base=self.api_base, model=self.model)
                self.reviewer = CriticalReviewer(openai_api_key=self.openai_api_key, api_base=self.api_base, model=self.model)
                logger.info(f"AI components initialized with model: {self.model}")
                if self.api_base:
                    logger.info(f"Using custom API endpoint: {self.api_base}")
            else:
                logger.warning("No API keys provided - analysis features will be limited")
            
            self.initialized = True
            logger.info("System initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise

    async def analyze_article(self, input_data: str, input_type: str = "auto") -> AnalysisResult:
        """
        Complete analysis workflow for a scientific article.
        
        Args:
            input_data: The article data (PDF path, URL, or raw text)
            input_type: Type of input ("pdf", "url", "text", or "auto" for detection)
            
        Returns:
            AnalysisResult: Complete analysis including classification, extraction, and review
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"Starting analysis of article: {input_data[:100]}...")
        
        try:
            # Step 1: Process the article to extract text content
            article_content = await self.processor.process_article(input_data, input_type)
            logger.info(f"Article processed successfully. Title: {article_content.title}")
            
            # Check if AI components are available
            if not self.classifier or not self.extractor or not self.reviewer:
                raise ValueError("Analysis requires OpenAI API key. Please provide one during initialization.")
            
            # Step 2: Classify the article into scientific category
            # Get reference articles for classification
            reference_articles = []
            for category in ScientificCategory:
                refs = self.vector_store.get_reference_articles(category)
                for ref in refs:
                    reference_articles.append(VectorStoreEntry(
                        id=ref.get('id', ''),
                        title=ref.get('title', ''),
                        abstract=ref.get('abstract', ''),
                        content=ref.get('content', ''),
                        category=category,
                        metadata=ref
                    ))
            
            classification_result = await self.classifier.classify_article(article_content, reference_articles)
            logger.info(f"Article classified as: {classification_result.category.value}")
            
            # Step 3: Extract structured information in JSON format
            extracted_info = await self.extractor.extract_information(article_content)
            logger.info("Information extracted successfully")
            
            # Step 4: Generate critical review
            review = await self.reviewer.generate_review(
                article_content, classification_result.category
            )
            logger.info("Critical review generated successfully")
            
            # Create comprehensive result
            result = AnalysisResult(
                article_content=article_content,
                classification=classification_result,
                extracted_info=extracted_info,
                review=review
            )
            
            logger.info("Complete analysis finished successfully!")
            return result
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise

    async def add_reference_article(self, input_data: str, category: ScientificCategory, 
                                  input_type: str = "auto") -> Dict[str, Any]:
        """
        Add a new reference article to the vector store.
        
        Args:
            input_data: The article data (PDF path, URL, or raw text)
            category: Scientific category of the article
            input_type: Type of input ("pdf", "url", "text", or "auto")
            
        Returns:
            Dict containing success status and article info
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            article_content = await self.processor.process_article(input_data, input_type)
            await self.vector_store.add_article(article_content, category)
            
            return {
                "success": True,
                "message": f"Reference article added successfully to {category.value}",
                "article_id": f"{category.value}_{article_content.title[:50]}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to add reference article: {e}"
            }

    async def search_similar_articles(self, query: str, category: Optional[ScientificCategory] = None,
                                    limit: int = 5) -> Dict[str, Any]:
        """
        Search for similar articles in the vector store.
        
        Args:
            query: Search query text
            category: Optional category filter
            limit: Maximum number of results
            
        Returns:
            Dict containing search results
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            results = await self.vector_store.search_similar(
                query, category=category, limit=limit
            )
            
            return {
                "success": True,
                "results": [
                    {
                        "title": result["title"],
                        "abstract": result["abstract"][:200] + "...",
                        "category": result["category"],
                        "similarity": result["similarity"]
                    }
                    for result in results
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Search failed: {e}"
            }

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and status.
        
        Returns:
            Dict containing system information
        """
        return {
            "initialized": self.initialized,
            "components": {
                "processor": "ArticleProcessor",
                "classifier": "ArticleClassifier",  
                "extractor": "InformationExtractor",
                "reviewer": "CriticalReviewer",
                "vector_store": "ChromaDB Vector Store",
                "embedding_model": "all-MiniLM-L6-v2"
            },
            "supported_categories": [cat.value for cat in ScientificCategory],
            "supported_inputs": ["PDF files", "URLs", "Raw text"],
            "vector_store_stats": self.vector_store.get_stats() if self.initialized else {}
        }

async def main():
    """
    Main function demonstrating the system usage.
    """
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv(dotenv_path=project_root / ".env", override=True)
    
    # Get configuration from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")  # Para Ollama: http://localhost:11434/v1
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default: gpt-4o-mini
    
    # Normalizar valores
    openai_api_key = openai_api_key.strip() if openai_api_key else None
    api_base = api_base.strip() if api_base else None
    
    # Fallback para Ollama: exige alguma chave, mesmo local
    if api_base and not openai_api_key:
        openai_api_key = "ollama"
    if openai_api_key == "ollama":
        api_base = api_base or "http://localhost:11434/v1"
        if model == "gpt-4o-mini":
            model = "llama3.2"
    
    print("🔬 Scientific Article Analysis System")
    print("=" * 50)
    
    # Initialize the analyzer with configuration
    analyzer = ScientificArticleAnalyzer(
        openai_api_key=openai_api_key,
        api_base=api_base,
        model=model
    )
    
    # Example usage
    print("\n📊 System Statistics:")
    stats = analyzer.get_system_stats()
    print(json.dumps(stats, indent=2))
    
    # Initialize the system
    print("\n🚀 Initializing system...")
    await analyzer.initialize()
    
    # Show configuration
    if api_base:
        if "localhost:11434" in api_base:
            print(f"\n✅ Usando Ollama: {api_base}")
        else:
            print(f"\n✅ Usando API customizada: {api_base}")
        print(f"📦 Modelo: {model}")
    elif openai_api_key:
        print(f"\n✅ Usando OpenAI API")
        print(f"📦 Modelo: {model}")
    else:
        print("\n⚠️  Nenhuma chave API configurada - funcionalidades limitadas")
    
    print("\n✅ System ready! You can now:")
    print("1. Analyze articles using analyzer.analyze_article()")
    print("2. Add reference articles using analyzer.add_reference_article()")
    print("3. Search similar articles using analyzer.search_similar_articles()")
    
    # Example analysis (uncomment to test with a real article)
    """
    print("\n🧪 Testing with sample article...")
    sample_text = '''
    Title: Machine Learning Approaches for Natural Language Processing
    
    Abstract: This paper presents a comprehensive study of machine learning techniques
    applied to natural language processing tasks. We investigate deep learning models
    including transformers and recurrent neural networks for text classification,
    sentiment analysis, and language generation.
    
    The main contribution of this work is a novel architecture that combines
    attention mechanisms with convolutional layers to improve performance on
    various NLP benchmarks. Our experiments show significant improvements
    over baseline methods.
    '''
    
    try:
        result = await analyzer.analyze_article(sample_text, "text")
        print("\n📋 Analysis Result:")
        print(f"Category: {result.classification.category.value}")
        print(f"Confidence: {result.classification.confidence:.2f}")
        print(f"Problem: {result.extracted_info.problem}")
        print(f"Review Summary: {result.review.summary[:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    """
    
    print("\n🎯 System is ready for use!")

if __name__ == "__main__":
    asyncio.run(main())