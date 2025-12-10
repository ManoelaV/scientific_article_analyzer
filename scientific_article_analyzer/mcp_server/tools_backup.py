from typing import Any, Dict, List
import json
import asyncio
from mcp.server.fastmcp import FastMCP
from mcp.server.models import Tool
from mcp.shared.exceptions import McpError
from pydantic import Field

from ..src import (
    ArticleProcessor,
    ArticleClassifier, 
    InformationExtractor,
    CriticalReviewer
)
from ..src.models import (
    AnalyzeArticleRequest,
    ClassifyArticleRequest,
    ExtractInfoRequest,
    GenerateReviewRequest,
    SearchSimilarRequest,
    AddReferenceArticleRequest,
    InputType,
    ScientificCategory
)
from ..vector_store import VectorStore


class MCPTools:
    """MCP tools for the scientific article analyzer."""
    
    def __init__(self, openai_api_key: str, vector_store_path: str = "./vector_store_db"):
        self.openai_api_key = openai_api_key
        self.vector_store = VectorStore(vector_store_path)
        
        # Initialize processors
        self.article_processor = ArticleProcessor()
        self.classifier = ArticleClassifier(openai_api_key)
        self.extractor = InformationExtractor(openai_api_key)
        self.reviewer = CriticalReviewer(openai_api_key)
    
    async def analyze_article(self, request: AnalyzeArticleRequest) -> Dict[str, Any]:
        """Complete article analysis including classification, extraction, and review."""
        try:
            # Process the article
            article = await self.article_processor.process_article(
                request.input_data, 
                request.input_type
            )
            
            result = {
                "article_info": {
                    "title": article.title,
                    "authors": article.authors,
                    "abstract": article.abstract,
                    "keywords": article.keywords
                }
            }
            
            # Classification
            if request.include_classification:
                reference_articles = await self.vector_store.get_reference_articles()
                classification = await self.classifier.classify_article(article, reference_articles)
                result["classification"] = {
                    "category": classification.category.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning
                }
            
            # Information extraction
            if request.include_extraction:
                extracted_info = await self.extractor.extract_information(article)
                result["extracted_info"] = {
                    "what problem does the article propose to solve?": extracted_info.what_problem_does_the_article_propose_to_solve,
                    "step by step on how to solve it": extracted_info.step_by_step_on_how_to_solve_it,
                    "conclusion": extracted_info.conclusion
                }
            
            # Critical review
            if request.include_review:
                # Use classification result if available, otherwise classify for review
                if request.include_classification:
                    category = classification.category
                else:
                    reference_articles = await self.vector_store.get_reference_articles()
                    temp_classification = await self.classifier.classify_article(article, reference_articles)
                    category = temp_classification.category
                
                review = await self.reviewer.generate_review(article, category)
                result["critical_review"] = {
                    "positive_aspects": review.positive_aspects,
                    "possible_flaws": review.possible_flaws,
                    "overall_assessment": review.overall_assessment,
                    "methodology_evaluation": review.methodology_evaluation,
                    "significance_rating": review.significance_rating,
                    "recommendations": review.recommendations
                }
            
            return {
                "status": "success",
                "data": result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to analyze article: {str(e)}"
            }
    
    async def classify_article(self, request: ClassifyArticleRequest) -> Dict[str, Any]:
        """Classify article into scientific category."""
        try:
            # Process the article
            article = await self.article_processor.process_article(
                request.input_data,
                request.input_type
            )
            
            # Get reference articles for classification
            reference_articles = await self.vector_store.get_reference_articles()
            
            # Classify the article
            classification = await self.classifier.classify_article(article, reference_articles)
            
            return {
                "status": "success",
                "data": {
                    "category": classification.category.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning,
                    "article_title": article.title
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to classify article: {str(e)}"
            }
    
    async def extract_article_info(self, request: ExtractInfoRequest) -> Dict[str, Any]:
        """Extract structured information from article."""
        try:
            # Process the article
            article = await self.article_processor.process_article(
                request.input_data,
                request.input_type
            )
            
            # Extract information
            extracted_info = await self.extractor.extract_information(article)
            
            return {
                "status": "success",
                "data": {
                    "what problem does the article propose to solve?": extracted_info.what_problem_does_the_article_propose_to_solve,
                    "step by step on how to solve it": extracted_info.step_by_step_on_how_to_solve_it,
                    "conclusion": extracted_info.conclusion,
                    "article_title": article.title
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to extract information: {str(e)}"
            }
    
    async def generate_article_review(self, request: GenerateReviewRequest) -> Dict[str, Any]:
        """Generate critical review of article."""
        try:
            # Process the article
            article = await self.article_processor.process_article(
                request.input_data,
                request.input_type
            )
            
            # Classify to determine review criteria
            reference_articles = await self.vector_store.get_reference_articles()
            classification = await self.classifier.classify_article(article, reference_articles)
            
            # Generate review
            review = await self.reviewer.generate_review(article, classification.category)
            
            return {
                "status": "success",
                "data": {
                    "positive_aspects": review.positive_aspects,
                    "possible_flaws": review.possible_flaws,
                    "overall_assessment": review.overall_assessment,
                    "methodology_evaluation": review.methodology_evaluation,
                    "significance_rating": review.significance_rating,
                    "recommendations": review.recommendations,
                    "article_title": article.title,
                    "classified_category": classification.category.value
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to generate review: {str(e)}"
            }
    
    async def search_similar_articles(self, request: SearchSimilarRequest) -> Dict[str, Any]:
        """Search for similar articles in the vector store."""
        try:
            # Search for similar articles
            results = await self.vector_store.search_similar(
                query_text=request.query_text,
                category=request.category,
                max_results=request.max_results,
                similarity_threshold=0.1  # Minimum similarity threshold
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "article_id": result.entry.id,
                    "title": result.entry.metadata.get("title", "Unknown"),
                    "authors": result.entry.metadata.get("authors", ""),
                    "category": result.entry.category.value,
                    "similarity_score": result.similarity_score,
                    "relevance_explanation": result.relevance_explanation,
                    "is_reference": result.entry.is_reference,
                    "abstract": result.entry.metadata.get("abstract", "")[:200] + "..." if result.entry.metadata.get("abstract") else ""
                })
            
            return {
                "status": "success",
                "data": {
                    "query": request.query_text,
                    "results_count": len(formatted_results),
                    "results": formatted_results
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to search similar articles: {str(e)}"
            }
    
    async def add_reference_article(self, request: AddReferenceArticleRequest) -> Dict[str, Any]:
        """Add a new reference article to the vector store."""
        try:
            # Process the article
            article = await self.article_processor.process_article(
                request.input_data,
                request.input_type
            )
            
            # Add to vector store
            article_id = await self.vector_store.add_article(
                article,
                request.category,
                is_reference=request.is_reference
            )
            
            return {
                "status": "success",
                "data": {
                    "article_id": article_id,
                    "title": article.title,
                    "category": request.category.value,
                    "is_reference": request.is_reference,
                    "message": f"Successfully added article '{article.title}' to the vector store"
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to add reference article: {str(e)}"
            }
    
    async def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            counts = await self.vector_store.get_article_count()
            
            total_articles = sum(cat_data["total_articles"] for cat_data in counts.values())
            total_references = sum(cat_data["reference_articles"] for cat_data in counts.values())
            
            return {
                "status": "success",
                "data": {
                    "total_articles": total_articles,
                    "total_reference_articles": total_references,
                    "by_category": counts,
                    "categories": [category.value for category in ScientificCategory]
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to get vector store statistics: {str(e)}"
            }


def create_mcp_tools(app: FastMCP, tools: MCPTools):
    """Register all MCP tools with the FastMCP app."""
    
    @app.tool()
    async def analyze_article(
        input_data: str = Field(description="Path to PDF, URL, or raw text of the article"),
        input_type: str = Field(description="Type of input: 'pdf', 'url', or 'text'"),
        include_classification: bool = Field(default=True, description="Include article classification"),
        include_extraction: bool = Field(default=True, description="Include information extraction"),
        include_review: bool = Field(default=True, description="Include critical review")
    ) -> str:
        """Perform complete analysis of a scientific article including classification, information extraction, and critical review."""
        
        try:
            input_type_enum = InputType(input_type.lower())
        except ValueError:
            return json.dumps({
                "status": "error",
                "error": f"Invalid input_type. Must be one of: {', '.join([t.value for t in InputType])}"
            })
        
        request = AnalyzeArticleRequest(
            input_data=input_data,
            input_type=input_type_enum,
            include_classification=include_classification,
            include_extraction=include_extraction,
            include_review=include_review
        )
        
        result = await tools.analyze_article(request)
        return json.dumps(result, indent=2)
    
    @app.tool()
    async def classify_article(
        input_data: str = Field(description="Path to PDF, URL, or raw text of the article"),
        input_type: str = Field(description="Type of input: 'pdf', 'url', or 'text'")
    ) -> str:
        """Classify a scientific article into one of the supported categories: Computer Science, Physics, or Biology."""
        
        try:
            input_type_enum = InputType(input_type.lower())
        except ValueError:
            return json.dumps({
                "status": "error",
                "error": f"Invalid input_type. Must be one of: {', '.join([t.value for t in InputType])}"
            })
        
        request = ClassifyArticleRequest(
            input_data=input_data,
            input_type=input_type_enum
        )
        
        result = await tools.classify_article(request)
        return json.dumps(result, indent=2)
    
    @app.tool()
    async def extract_article_info(
        input_data: str = Field(description="Path to PDF, URL, or raw text of the article"),
        input_type: str = Field(description="Type of input: 'pdf', 'url', or 'text'")
    ) -> str:
        """Extract structured information from a scientific article in JSON format: problem statement, solution steps, and conclusion."""
        
        try:
            input_type_enum = InputType(input_type.lower())
        except ValueError:
            return json.dumps({
                "status": "error",
                "error": f"Invalid input_type. Must be one of: {', '.join([t.value for t in InputType])}"
            })
        
        request = ExtractInfoRequest(
            input_data=input_data,
            input_type=input_type_enum
        )
        
        result = await tools.extract_article_info(request)
        return json.dumps(result, indent=2)
    
    @app.tool()
    async def generate_article_review(
        input_data: str = Field(description="Path to PDF, URL, or raw text of the article"),
        input_type: str = Field(description="Type of input: 'pdf', 'url', or 'text'")
    ) -> str:
        """Generate a comprehensive critical review of a scientific article, highlighting positive aspects and potential flaws."""
        
        try:
            input_type_enum = InputType(input_type.lower())
        except ValueError:
            return json.dumps({
                "status": "error",
                "error": f"Invalid input_type. Must be one of: {', '.join([t.value for t in InputType])}"
            })
        
        request = GenerateReviewRequest(
            input_data=input_data,
            input_type=input_type_enum
        )
        
        result = await tools.generate_article_review(request)
        return json.dumps(result, indent=2)
    
    @app.tool()
    async def search_similar_articles(
        query_text: str = Field(description="Text to search for similar articles"),
        category: str = Field(default="", description="Limit search to category: 'Computer Science', 'Physics', 'Biology', or empty for all"),
        max_results: int = Field(default=5, description="Maximum number of results to return")
    ) -> str:
        """Search for articles similar to the provided text query in the vector store knowledge base."""
        
        category_enum = None
        if category:
            try:
                # Map category string to enum
                category_map = {
                    "computer science": ScientificCategory.COMPUTER_SCIENCE,
                    "physics": ScientificCategory.PHYSICS,
                    "biology": ScientificCategory.BIOLOGY
                }
                category_enum = category_map.get(category.lower())
                if not category_enum:
                    return json.dumps({
                        "status": "error",
                        "error": f"Invalid category. Must be one of: {', '.join(category_map.keys())}"
                    })
            except ValueError:
                return json.dumps({
                    "status": "error",
                    "error": f"Invalid category. Must be one of: Computer Science, Physics, Biology"
                })
        
        request = SearchSimilarRequest(
            query_text=query_text,
            category=category_enum,
            max_results=max_results
        )
        
        result = await tools.search_similar_articles(request)
        return json.dumps(result, indent=2)
    
    @app.tool()
    async def add_reference_article(
        input_data: str = Field(description="Path to PDF, URL, or raw text of the article"),
        input_type: str = Field(description="Type of input: 'pdf', 'url', or 'text'"),
        category: str = Field(description="Scientific category: 'Computer Science', 'Physics', or 'Biology'"),
        is_reference: bool = Field(default=True, description="Mark as reference article for classification training")
    ) -> str:
        """Add a new reference article to the vector store knowledge base for improving classification accuracy."""
        
        try:
            input_type_enum = InputType(input_type.lower())
        except ValueError:
            return json.dumps({
                "status": "error",
                "error": f"Invalid input_type. Must be one of: {', '.join([t.value for t in InputType])}"
            })
        
        try:
            category_map = {
                "computer science": ScientificCategory.COMPUTER_SCIENCE,
                "physics": ScientificCategory.PHYSICS,
                "biology": ScientificCategory.BIOLOGY
            }
            category_enum = category_map.get(category.lower())
            if not category_enum:
                return json.dumps({
                    "status": "error",
                    "error": f"Invalid category. Must be one of: {', '.join(category_map.keys())}"
                })
        except ValueError:
            return json.dumps({
                "status": "error",
                "error": f"Invalid category. Must be one of: Computer Science, Physics, Biology"
            })
        
        request = AddReferenceArticleRequest(
            input_data=input_data,
            input_type=input_type_enum,
            category=category_enum,
            is_reference=is_reference
        )
        
        result = await tools.add_reference_article(request)
        return json.dumps(result, indent=2)
    
    @app.tool()
    async def get_vector_store_stats() -> str:
        """Get statistics about the articles stored in the vector store knowledge base."""
        result = await tools.get_vector_store_stats()
        return json.dumps(result, indent=2)