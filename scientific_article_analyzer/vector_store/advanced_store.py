"""
Advanced Vector Store Implementation with Chunking Strategy

This module implements a production-ready vector store with:
- Sophisticated text chunking with overlap
- Multiple embedding providers (OpenAI, HuggingFace, local)
- FAISS backend for high-performance similarity search
- Centroid-based classification with few-shot retrieval
- Robust error handling and metadata management
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
from pathlib import Path

# Vector store backends
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    
# Embedding providers
from sentence_transformers import SentenceTransformer
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.models import ScientificCategory, ArticleContent

logger = logging.getLogger(__name__)

class EmbeddingProvider(str, Enum):
    """Available embedding providers."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"

class ChunkingStrategy(str, Enum):
    """Text chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    chunk_size: int = 1000  # tokens
    overlap: int = 200      # tokens
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    
    def __post_init__(self):
        """Validate chunking configuration."""
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size cannot be larger than chunk_size")

@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    content: str
    chunk_id: str
    document_id: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunking_config: ChunkingConfig = None
    storage_path: str = "./data/advanced_vector_store"
    use_faiss: bool = True
    similarity_threshold: float = 0.7
    
    def __post_init__(self):
        if self.chunking_config is None:
            self.chunking_config = ChunkingConfig()

class TextChunker:
    """Advanced text chunking with multiple strategies."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        # Initialize tokenizer for accurate token counting
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not available, using approximate token counting")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: 1 token ≈ 4 characters for English
            return len(text) // 4
    
    def chunk_text(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Chunk text according to configured strategy."""
        if self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text, document_id)
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(text, document_id)
        elif self.config.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_paragraph(text, document_id)
        elif self.config.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_sentence(text, document_id)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")
    
    def _chunk_fixed_size(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Fixed-size chunking with overlap.
        
        Justification for window size and overlap:
        - Chunk size of 1000 tokens: Optimal balance between context preservation 
          and processing efficiency. Large enough to maintain semantic coherence 
          but small enough for fast embedding generation.
        - Overlap of 200 tokens (20%): Ensures continuity across chunks and 
          prevents loss of context at boundaries. 20% overlap is empirically 
          proven to maintain semantic relationships while minimizing redundancy.
        """
        chunks = []
        tokens_per_char = 4  # Approximate for English
        
        if self.tokenizer:
            # Use accurate tokenization
            tokens = self.tokenizer.encode(text)
            chunk_size_tokens = self.config.chunk_size
            overlap_tokens = self.config.overlap
            
            for i in range(0, len(tokens), chunk_size_tokens - overlap_tokens):
                chunk_tokens = tokens[i:i + chunk_size_tokens]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                if len(chunk_tokens) < self.config.min_chunk_size:
                    continue
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    document_id=document_id,
                    start_pos=i,
                    end_pos=i + len(chunk_tokens),
                    metadata={
                        "token_count": len(chunk_tokens),
                        "char_start": 0,  # Would need mapping for exact char positions
                        "char_end": len(chunk_text),
                        "chunk_strategy": "fixed_size"
                    }
                )
                chunks.append(chunk)
        else:
            # Fallback to character-based chunking
            chunk_size_chars = self.config.chunk_size * tokens_per_char
            overlap_chars = self.config.overlap * tokens_per_char
            
            for i in range(0, len(text), chunk_size_chars - overlap_chars):
                chunk_text = text[i:i + chunk_size_chars]
                
                if len(chunk_text) < self.config.min_chunk_size * tokens_per_char:
                    continue
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    document_id=document_id,
                    start_pos=i,
                    end_pos=i + len(chunk_text),
                    metadata={
                        "token_count": self.count_tokens(chunk_text),
                        "char_start": i,
                        "char_end": i + len(chunk_text),
                        "chunk_strategy": "fixed_size"
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_semantic(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Semantic chunking based on paragraph boundaries."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            combined = current_chunk + ("\n\n" if current_chunk else "") + para
            token_count = self.count_tokens(combined)
            
            if token_count > self.config.chunk_size and current_chunk:
                # Create chunk with current content
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_id=f"{document_id}_semantic_{len(chunks)}",
                    document_id=document_id,
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk),
                    metadata={
                        "token_count": self.count_tokens(current_chunk),
                        "chunk_strategy": "semantic",
                        "paragraph_count": current_chunk.count('\n\n') + 1
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.config.overlap > 0:
                    overlap_text = current_chunk[-self.config.overlap * 4:]  # Approx chars
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
                current_start = len(text) - len(current_chunk)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                chunk_id=f"{document_id}_semantic_{len(chunks)}",
                document_id=document_id,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                metadata={
                    "token_count": self.count_tokens(current_chunk),
                    "chunk_strategy": "semantic",
                    "paragraph_count": current_chunk.count('\n\n') + 1
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_paragraph(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Chunk by paragraphs with size limits."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        
        for i, para in enumerate(paragraphs):
            token_count = self.count_tokens(para)
            
            # Split large paragraphs
            if token_count > self.config.max_chunk_size:
                sub_chunks = self._chunk_fixed_size(para, f"{document_id}_para_{i}")
                chunks.extend(sub_chunks)
            elif token_count >= self.config.min_chunk_size:
                chunk = DocumentChunk(
                    content=para,
                    chunk_id=f"{document_id}_para_{i}",
                    document_id=document_id,
                    start_pos=0,  # Would need full text indexing for exact positions
                    end_pos=len(para),
                    metadata={
                        "token_count": token_count,
                        "chunk_strategy": "paragraph",
                        "paragraph_index": i
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_sentence(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Chunk by sentences with size limits."""
        # Simple sentence splitting (could be enhanced with NLP libraries)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        chunks = []
        current_chunk = ""
        sentence_start_idx = 0
        
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
            
            sentence += "."  # Re-add period
            combined = current_chunk + (" " if current_chunk else "") + sentence
            token_count = self.count_tokens(combined)
            
            if token_count > self.config.chunk_size and current_chunk:
                # Create chunk
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_id=f"{document_id}_sent_{len(chunks)}",
                    document_id=document_id,
                    start_pos=sentence_start_idx,
                    end_pos=sentence_start_idx + len(current_chunk),
                    metadata={
                        "token_count": self.count_tokens(current_chunk),
                        "chunk_strategy": "sentence",
                        "sentence_count": current_chunk.count('.') if current_chunk.count('.') > 0 else 1
                    }
                )
                chunks.append(chunk)
                
                current_chunk = sentence
                sentence_start_idx = i
            else:
                current_chunk = combined
        
        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                chunk_id=f"{document_id}_sent_{len(chunks)}",
                document_id=document_id,
                start_pos=sentence_start_idx,
                end_pos=sentence_start_idx + len(current_chunk),
                metadata={
                    "token_count": self.count_tokens(current_chunk),
                    "chunk_strategy": "sentence",
                    "sentence_count": current_chunk.count('.')
                }
            )
            chunks.append(chunk)
        
        return chunks

class EmbeddingManager:
    """Advanced embedding management with multiple providers."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.provider = config.embedding_provider
        self.model_name = config.embedding_model
        self.dimension = config.embedding_dimension
        
        # Initialize embedding model based on provider
        if self.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        elif self.provider == EmbeddingProvider.OPENAI and OPENAI_AVAILABLE:
            # OpenAI embeddings will be called via API
            self.model = None
            self.dimension = 1536  # text-embedding-ada-002 dimension
        else:
            raise ValueError(f"Embedding provider {self.provider} not available or supported")
    
    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text."""
        try:
            if self.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
                return self._create_sentence_transformer_embedding(text)
            elif self.provider == EmbeddingProvider.OPENAI:
                return await self._create_openai_embedding(text)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            # Return zero embedding as fallback
            return [0.0] * self.dimension
    
    def _create_sentence_transformer_embedding(self, text: str) -> List[float]:
        """Create embedding using Sentence Transformers."""
        if not text.strip():
            return [0.0] * self.dimension
        
        # Clean and preprocess text
        clean_text = self._preprocess_text(text)
        embedding = self.model.encode(clean_text)
        return embedding.tolist()
    
    async def _create_openai_embedding(self, text: str) -> List[float]:
        """Create embedding using OpenAI API."""
        if not text.strip():
            return [0.0] * self.dimension
        
        try:
            clean_text = self._preprocess_text(text)
            response = await openai.Embedding.acreate(
                input=clean_text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return [0.0] * self.dimension
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (most models have token limits)
        max_chars = 8000  # Conservative limit
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        return text

class AdvancedVectorStore:
    """
    Advanced vector store with FAISS backend and sophisticated chunking.
    
    Features:
    - Multiple embedding providers
    - Intelligent text chunking with configurable strategies
    - FAISS for high-performance similarity search
    - Centroid-based classification
    - Robust error handling and persistence
    """
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(config)
        self.chunker = TextChunker(config.chunking_config)
        
        # Initialize FAISS index
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.category_centroids: Dict[str, np.ndarray] = {}
        
        # Metadata storage
        self.metadata_file = self.storage_path / "metadata.json"
        self.chunks_file = self.storage_path / "chunks.json"
        self.index_file = self.storage_path / "faiss_index.bin"
        self.centroids_file = self.storage_path / "centroids.pkl"
        
        # Load existing data
        self._load_data()
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index for similarity search."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, falling back to basic similarity search")
            return
        
        dimension = self.embedding_manager.dimension
        
        # Use IndexFlatIP (Inner Product) for cosine similarity
        # Normalize embeddings for cosine similarity with inner product
        self.index = faiss.IndexFlatIP(dimension)
        
        if self.chunks and all(chunk.embedding for chunk in self.chunks):
            # Add existing embeddings to index
            embeddings = np.array([chunk.embedding for chunk in self.chunks if chunk.embedding])
            if len(embeddings) > 0:
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings.astype('float32'))
    
    def _load_data(self):
        """Load persisted data."""
        try:
            # Load chunks
            if self.chunks_file.exists():
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                    self.chunks = [DocumentChunk(**chunk_data) for chunk_data in chunks_data]
            
            # Load centroids
            if self.centroids_file.exists():
                with open(self.centroids_file, 'rb') as f:
                    self.category_centroids = pickle.load(f)
            
            # Initialize FAISS index
            self._initialize_faiss_index()
            
            # Load FAISS index if it exists
            if self.index_file.exists() and FAISS_AVAILABLE:
                try:
                    self.index = faiss.read_index(str(self.index_file))
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}")
                    self._initialize_faiss_index()
            
            logger.info(f"Loaded {len(self.chunks)} chunks and {len(self.category_centroids)} centroids")
        
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self._initialize_faiss_index()
    
    def _save_data(self):
        """Persist data to disk."""
        try:
            # Save chunks
            chunks_data = [asdict(chunk) for chunk in self.chunks]
            with open(self.chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            # Save centroids
            with open(self.centroids_file, 'wb') as f:
                pickle.dump(self.category_centroids, f)
            
            # Save FAISS index
            if self.index and FAISS_AVAILABLE:
                faiss.write_index(self.index, str(self.index_file))
            
            logger.info(f"Saved {len(self.chunks)} chunks and {len(self.category_centroids)} centroids")
        
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    async def add_document(self, article: ArticleContent, category: ScientificCategory) -> bool:
        """Add document to vector store with chunking."""
        try:
            document_id = f"{category.value}_{len([c for c in self.chunks if c.metadata.get('category') == category.value])}"
            
            # Create chunks
            chunks = self.chunker.chunk_text(article.full_text, document_id)
            
            # Generate embeddings for chunks
            for chunk in chunks:
                embedding = await self.embedding_manager.create_embedding(chunk.content)
                chunk.embedding = embedding
                chunk.metadata.update({
                    'category': category.value,
                    'title': article.title,
                    'abstract': article.abstract,
                    'authors': article.authors or [],
                    'keywords': article.keywords or []
                })
            
            # Add chunks to store
            self.chunks.extend(chunks)
            
            # Update FAISS index
            if FAISS_AVAILABLE and self.index is not None:
                new_embeddings = np.array([chunk.embedding for chunk in chunks])
                faiss.normalize_L2(new_embeddings)
                self.index.add(new_embeddings.astype('float32'))
            
            # Update category centroids
            await self._update_centroids()
            
            # Persist data
            self._save_data()
            
            logger.info(f"Added document with {len(chunks)} chunks to category {category.value}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    async def _update_centroids(self):
        """Update category centroids for classification."""
        category_embeddings = {}
        
        for chunk in self.chunks:
            if not chunk.embedding:
                continue
            
            category = chunk.metadata.get('category')
            if category:
                if category not in category_embeddings:
                    category_embeddings[category] = []
                category_embeddings[category].append(chunk.embedding)
        
        # Calculate centroids
        for category, embeddings in category_embeddings.items():
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                self.category_centroids[category] = centroid
        
        logger.info(f"Updated centroids for {len(self.category_centroids)} categories")
    
    async def search_similar(self, query: str, category: Optional[ScientificCategory] = None,
                           limit: int = 5, similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using FAISS or fallback method."""
        try:
            if similarity_threshold is None:
                similarity_threshold = self.config.similarity_threshold
            
            # Create query embedding
            query_embedding = await self.embedding_manager.create_embedding(query)
            query_array = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_array)
            
            results = []
            
            if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
                # Use FAISS for fast search
                scores, indices = self.index.search(query_array, min(limit * 2, self.index.ntotal))
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.chunks) and score >= similarity_threshold:
                        chunk = self.chunks[idx]
                        
                        # Filter by category if specified
                        if category and chunk.metadata.get('category') != category.value:
                            continue
                        
                        results.append({
                            'chunk': chunk,
                            'similarity': float(score),
                            'content': chunk.content,
                            'metadata': chunk.metadata
                        })
                        
                        if len(results) >= limit:
                            break
            
            else:
                # Fallback to manual cosine similarity
                query_np = np.array(query_embedding)
                
                for chunk in self.chunks:
                    if not chunk.embedding:
                        continue
                    
                    # Filter by category if specified
                    if category and chunk.metadata.get('category') != category.value:
                        continue
                    
                    # Calculate cosine similarity
                    chunk_np = np.array(chunk.embedding)
                    similarity = np.dot(query_np, chunk_np) / (np.linalg.norm(query_np) * np.linalg.norm(chunk_np))
                    
                    if similarity >= similarity_threshold:
                        results.append({
                            'chunk': chunk,
                            'similarity': float(similarity),
                            'content': chunk.content,
                            'metadata': chunk.metadata
                        })
                
                # Sort by similarity
                results.sort(key=lambda x: x['similarity'], reverse=True)
                results = results[:limit]
            
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def classify_by_centroid(self, text: str) -> Dict[str, Any]:
        """
        Classify text using centroid-based similarity.
        
        This method combines similarity search to category centroids with
        few-shot retrieval for enhanced classification accuracy.
        """
        try:
            # Create embedding for input text
            text_embedding = await self.embedding_manager.create_embedding(text)
            text_np = np.array(text_embedding)
            
            centroid_similarities = {}
            
            # Calculate similarity to each category centroid
            for category, centroid in self.category_centroids.items():
                similarity = np.dot(text_np, centroid) / (np.linalg.norm(text_np) * np.linalg.norm(centroid))
                centroid_similarities[category] = float(similarity)
            
            if not centroid_similarities:
                return {"category": "Unknown", "confidence": 0.0, "method": "centroid"}
            
            # Get best matching category
            best_category = max(centroid_similarities.items(), key=lambda x: x[1])
            
            # Few-shot retrieval for verification
            # Find most similar chunks for context
            similar_chunks = await self.search_similar(text, limit=3)
            
            # Analyze category distribution in similar chunks
            category_votes = {}
            total_similarity = 0
            
            for result in similar_chunks:
                chunk_category = result['metadata'].get('category')
                similarity = result['similarity']
                
                if chunk_category:
                    if chunk_category not in category_votes:
                        category_votes[chunk_category] = 0
                    category_votes[chunk_category] += similarity
                    total_similarity += similarity
            
            # Normalize votes
            if total_similarity > 0:
                for cat in category_votes:
                    category_votes[cat] /= total_similarity
            
            # Combine centroid and retrieval methods
            final_scores = {}
            for category in ScientificCategory:
                centroid_score = centroid_similarities.get(category.value, 0.0)
                retrieval_score = category_votes.get(category.value, 0.0)
                
                # Weighted combination (70% centroid, 30% retrieval)
                final_score = 0.7 * centroid_score + 0.3 * retrieval_score
                final_scores[category.value] = final_score
            
            # Get final classification
            best_final = max(final_scores.items(), key=lambda x: x[1])
            
            return {
                "category": best_final[0],
                "confidence": best_final[1],
                "method": "centroid_plus_retrieval",
                "centroid_scores": centroid_similarities,
                "retrieval_votes": category_votes,
                "final_scores": final_scores,
                "similar_chunks_count": len(similar_chunks)
            }
        
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {"category": "Unknown", "confidence": 0.0, "method": "error", "error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        category_stats = {}
        
        for category in ScientificCategory:
            chunks_in_category = [c for c in self.chunks if c.metadata.get('category') == category.value]
            documents_in_category = len(set(c.document_id for c in chunks_in_category))
            
            category_stats[category.value] = {
                'document_count': documents_in_category,
                'chunk_count': len(chunks_in_category),
                'avg_chunk_size': np.mean([c.metadata.get('token_count', 0) for c in chunks_in_category]) if chunks_in_category else 0,
                'has_centroid': category.value in self.category_centroids
            }
        
        return {
            'total_chunks': len(self.chunks),
            'total_documents': len(set(c.document_id for c in self.chunks)),
            'embedding_dimension': self.embedding_manager.dimension,
            'embedding_provider': self.config.embedding_provider.value,
            'chunking_strategy': self.config.chunking_config.strategy.value,
            'chunk_size': self.config.chunking_config.chunk_size,
            'overlap': self.config.chunking_config.overlap,
            'faiss_available': FAISS_AVAILABLE,
            'index_size': self.index.ntotal if self.index else 0,
            'category_statistics': category_stats,
            'storage_path': str(self.storage_path)
        }
    
    async def initialize_with_sample_articles(self):
        """Initialize with sample articles for testing."""
        # This would be called from the main initialization
        # Implementation would add sample articles to each category
        pass