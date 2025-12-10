import os
import json
import asyncio
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from ..src.models import (
    ScientificCategory, 
    ArticleContent, 
    VectorStoreEntry, 
    SimilaritySearchResult
)
from .embeddings import EmbeddingManager


class VectorStore:
    """Simple vector store for scientific articles using numpy and sklearn."""
    
    def __init__(self, store_path: str = "./vector_store_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.store_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(embedding_model)
        
        # Collection names for different categories
        self.collections = {
            ScientificCategory.COMPUTER_SCIENCE: "computer_science_articles",
            ScientificCategory.PHYSICS: "physics_articles", 
            ScientificCategory.BIOLOGY: "biology_articles"
        }
        
        # Initialize collections
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections for each scientific category."""
        for category, collection_name in self.collections.items():
            try:
                self.client.get_collection(collection_name)
            except:
                # Create collection if it doesn't exist
                self.client.create_collection(
                    name=collection_name,
                    metadata={"category": category.value}
                )
    
    async def add_article(self, article: ArticleContent, category: ScientificCategory, is_reference: bool = False) -> str:
        """Add an article to the vector store."""
        
        # Generate unique ID
        article_id = self._generate_article_id(article, category)
        
        # Prepare text for embedding
        article_text = self._prepare_article_text(article)
        
        # Create embeddings for chunks
        chunked_embeddings = self.embedding_manager.create_chunked_embeddings(article_text)
        
        if not chunked_embeddings:
            raise ValueError("Failed to create embeddings for article")
        
        # Get the appropriate collection
        collection = self.client.get_collection(self.collections[category])
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk_data in enumerate(chunked_embeddings):
            chunk_id = f"{article_id}_chunk_{i}"
            ids.append(chunk_id)
            embeddings.append(chunk_data["embedding"])
            documents.append(chunk_data["text"])
            metadatas.append({
                "article_id": article_id,
                "title": article.title,
                "authors": ", ".join(article.authors),
                "keywords": ", ".join(article.keywords),
                "category": category.value,
                "is_reference": is_reference,
                "chunk_id": chunk_data["chunk_id"],
                "token_count": chunk_data["token_count"],
                "added_date": datetime.now().isoformat(),
                "abstract": article.abstract[:500] if article.abstract else "",  # Limit length
                "publication_date": article.publication_date or "",
                "journal": article.journal or "",
                "doi": article.doi or ""
            })
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return article_id
    
    async def search_similar(self, query_text: str, category: Optional[ScientificCategory] = None, 
                           max_results: int = 5, similarity_threshold: float = 0.0) -> List[SimilaritySearchResult]:
        """Search for similar articles in the vector store."""
        
        # Create embedding for query
        query_embedding = self.embedding_manager.create_embedding(query_text)
        
        results = []
        categories_to_search = [category] if category else list(ScientificCategory)
        
        for cat in categories_to_search:
            collection = self.client.get_collection(self.collections[cat])
            
            try:
                # Search in ChromaDB
                search_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Process results
                for i, (document, metadata, distance) in enumerate(zip(
                    search_results["documents"][0],
                    search_results["metadatas"][0], 
                    search_results["distances"][0]
                )):
                    # Convert distance to similarity (ChromaDB uses cosine distance)
                    similarity = 1.0 - distance
                    
                    if similarity >= similarity_threshold:
                        # Create VectorStoreEntry
                        entry = VectorStoreEntry(
                            id=metadata["article_id"],
                            content=document,
                            embedding=query_embedding,  # We don't store the actual embedding in metadata
                            metadata=metadata,
                            category=ScientificCategory(metadata["category"]),
                            is_reference=metadata.get("is_reference", False)
                        )
                        
                        # Generate relevance explanation
                        relevance_explanation = self._generate_relevance_explanation(
                            query_text, document, metadata, similarity
                        )
                        
                        result = SimilaritySearchResult(
                            entry=entry,
                            similarity_score=similarity,
                            relevance_explanation=relevance_explanation
                        )
                        
                        results.append(result)
            
            except Exception as e:
                print(f"Error searching in {cat.value} collection: {e}")
                continue
        
        # Sort all results by similarity
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:max_results]
    
    async def get_reference_articles(self, category: Optional[ScientificCategory] = None) -> List[VectorStoreEntry]:
        """Get all reference articles from the store."""
        results = []
        categories_to_search = [category] if category else list(ScientificCategory)
        
        for cat in categories_to_search:
            collection = self.client.get_collection(self.collections[cat])
            
            try:
                # Query for reference articles
                search_results = collection.get(
                    where={"is_reference": True},
                    include=["documents", "metadatas"]
                )
                
                # Group by article_id to avoid duplicates from chunking
                articles_by_id = {}
                
                for document, metadata in zip(search_results["documents"], search_results["metadatas"]):
                    article_id = metadata["article_id"]
                    
                    if article_id not in articles_by_id:
                        articles_by_id[article_id] = {
                            "documents": [],
                            "metadata": metadata
                        }
                    
                    articles_by_id[article_id]["documents"].append(document)
                
                # Create VectorStoreEntry for each article
                for article_id, article_data in articles_by_id.items():
                    # Combine all chunks into full content
                    full_content = "\n\n".join(article_data["documents"])
                    
                    # Create embedding for full content
                    embedding = self.embedding_manager.create_embedding(full_content)
                    
                    entry = VectorStoreEntry(
                        id=article_id,
                        content=full_content,
                        embedding=embedding,
                        metadata=article_data["metadata"],
                        category=ScientificCategory(article_data["metadata"]["category"]),
                        is_reference=True
                    )
                    
                    results.append(entry)
            
            except Exception as e:
                print(f"Error getting reference articles from {cat.value}: {e}")
                continue
        
        return results
    
    async def get_article_count(self, category: Optional[ScientificCategory] = None) -> Dict[str, int]:
        """Get count of articles in the store."""
        counts = {}
        categories_to_count = [category] if category else list(ScientificCategory)
        
        for cat in categories_to_count:
            collection = self.client.get_collection(self.collections[cat])
            
            try:
                # Get all articles
                all_results = collection.get(include=["metadatas"])
                
                # Count unique articles (not chunks)
                unique_articles = set()
                reference_count = 0
                
                for metadata in all_results["metadatas"]:
                    unique_articles.add(metadata["article_id"])
                    if metadata.get("is_reference", False):
                        reference_count += 1
                
                counts[cat.value] = {
                    "total_articles": len(unique_articles),
                    "reference_articles": len(set(meta["article_id"] for meta in all_results["metadatas"] 
                                                if meta.get("is_reference", False))),
                    "total_chunks": len(all_results["metadatas"])
                }
                
            except Exception as e:
                print(f"Error counting articles in {cat.value}: {e}")
                counts[cat.value] = {"total_articles": 0, "reference_articles": 0, "total_chunks": 0}
        
        return counts
    
    async def delete_article(self, article_id: str, category: ScientificCategory) -> bool:
        """Delete an article from the vector store."""
        try:
            collection = self.client.get_collection(self.collections[category])
            
            # Find all chunks for this article
            results = collection.get(
                where={"article_id": article_id},
                include=["documents", "metadatas"]
            )
            
            if not results["ids"]:
                return False
            
            # Delete all chunks
            collection.delete(ids=results["ids"])
            return True
            
        except Exception as e:
            print(f"Error deleting article {article_id}: {e}")
            return False
    
    def _generate_article_id(self, article: ArticleContent, category: ScientificCategory) -> str:
        """Generate a unique ID for an article."""
        import hashlib
        
        # Create hash from title and authors
        content = f"{article.title}_{'-'.join(article.authors)}_{category.value}"
        hash_object = hashlib.md5(content.encode())
        return f"{category.value.lower().replace(' ', '_')}_{hash_object.hexdigest()[:8]}"
    
    def _prepare_article_text(self, article: ArticleContent) -> str:
        """Prepare article text for embedding."""
        parts = []
        
        if article.title:
            parts.append(f"Title: {article.title}")
        
        if article.abstract:
            parts.append(f"Abstract: {article.abstract}")
        
        if article.authors:
            parts.append(f"Authors: {', '.join(article.authors)}")
        
        if article.keywords:
            parts.append(f"Keywords: {', '.join(article.keywords)}")
        
        if article.full_text:
            parts.append(f"Full Text: {article.full_text}")
        
        return "\n\n".join(parts)
    
    def _generate_relevance_explanation(self, query: str, document: str, metadata: Dict, similarity: float) -> str:
        """Generate an explanation of why this document is relevant."""
        
        explanation_parts = [
            f"Similarity score: {similarity:.3f}",
            f"Article: '{metadata.get('title', 'Unknown')}'",
            f"Category: {metadata.get('category', 'Unknown')}"
        ]
        
        if metadata.get("keywords"):
            explanation_parts.append(f"Keywords: {metadata['keywords'][:100]}...")
        
        if metadata.get("is_reference"):
            explanation_parts.append("This is a reference article in the knowledge base")
        
        return " | ".join(explanation_parts)
    
    async def initialize_with_sample_articles(self):
        """Initialize the vector store with sample reference articles."""
        
        # This will be called by the initialization script
        # For now, we'll create placeholder entries
        
        sample_articles = {
            ScientificCategory.COMPUTER_SCIENCE: [
                {
                    "title": "Deep Learning for Natural Language Processing: A Comprehensive Survey",
                    "abstract": "This survey provides a comprehensive overview of deep learning techniques applied to natural language processing tasks, covering neural architectures, training methodologies, and recent advances in transformer models.",
                    "authors": ["John Smith", "Jane Doe"],
                    "keywords": ["deep learning", "natural language processing", "neural networks", "transformers"]
                },
                {
                    "title": "Quantum Computing Algorithms for Optimization Problems", 
                    "abstract": "We present novel quantum algorithms for solving complex optimization problems, demonstrating quantum advantage in specific computational scenarios.",
                    "authors": ["Alice Johnson", "Bob Wilson"],
                    "keywords": ["quantum computing", "optimization", "algorithms", "quantum advantage"]
                },
                {
                    "title": "Machine Learning Security: Adversarial Attacks and Defenses",
                    "abstract": "An analysis of security vulnerabilities in machine learning systems, with focus on adversarial attacks and corresponding defense mechanisms.", 
                    "authors": ["Carol Brown", "David Lee"],
                    "keywords": ["machine learning", "security", "adversarial attacks", "robustness"]
                }
            ],
            ScientificCategory.PHYSICS: [
                {
                    "title": "Quantum Entanglement in Many-Body Systems",
                    "abstract": "Investigation of quantum entanglement properties in complex many-body quantum systems, with applications to quantum information and condensed matter physics.",
                    "authors": ["Eva Martinez", "Frank Chen"],
                    "keywords": ["quantum entanglement", "many-body systems", "quantum information", "condensed matter"]
                },
                {
                    "title": "Dark Matter Detection Using Gravitational Lensing",
                    "abstract": "Novel approaches to dark matter detection through analysis of gravitational lensing effects in large-scale astronomical surveys.",
                    "authors": ["Grace Taylor", "Henry Kim"],
                    "keywords": ["dark matter", "gravitational lensing", "cosmology", "astrophysics"]
                },
                {
                    "title": "Superconductivity in High-Temperature Materials",
                    "abstract": "Experimental and theoretical studies of superconducting properties in novel high-temperature superconductor materials.",
                    "authors": ["Ivy Anderson", "Jack Thompson"],
                    "keywords": ["superconductivity", "high-temperature", "materials science", "condensed matter"]
                }
            ],
            ScientificCategory.BIOLOGY: [
                {
                    "title": "CRISPR Gene Editing: Applications and Ethical Considerations",
                    "abstract": "Comprehensive review of CRISPR-Cas9 gene editing technology, its applications in medicine and agriculture, and associated ethical implications.",
                    "authors": ["Kelly Rodriguez", "Luis Garcia"],
                    "keywords": ["CRISPR", "gene editing", "biotechnology", "ethics"]
                },
                {
                    "title": "Biodiversity Loss in Tropical Rainforests: Causes and Conservation",
                    "abstract": "Analysis of biodiversity decline in tropical rainforest ecosystems, examining anthropogenic causes and proposing conservation strategies.",
                    "authors": ["Maria Gonzalez", "Nathan White"],
                    "keywords": ["biodiversity", "tropical rainforests", "conservation", "ecology"]
                },
                {
                    "title": "Protein Folding Mechanisms in Neurodegenerative Diseases",
                    "abstract": "Investigation of protein misfolding mechanisms in Alzheimer's and Parkinson's diseases, with implications for therapeutic development.",
                    "authors": ["Olivia Davis", "Peter Miller"],
                    "keywords": ["protein folding", "neurodegenerative diseases", "Alzheimer's", "Parkinson's"]
                }
            ]
        }
        
        for category, articles in sample_articles.items():
            for article_data in articles:
                article = ArticleContent(
                    title=article_data["title"],
                    abstract=article_data["abstract"],
                    full_text=f"{article_data['title']}\n\n{article_data['abstract']}\n\nThis is a sample reference article for the {category.value} category.",
                    authors=article_data["authors"],
                    keywords=article_data["keywords"]
                )
                
                try:
                    await self.add_article(article, category, is_reference=True)
                    print(f"Added reference article: {article.title}")
                except Exception as e:
                    print(f"Error adding article {article.title}: {e}")