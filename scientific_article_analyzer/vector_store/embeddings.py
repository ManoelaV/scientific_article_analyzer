import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import tiktoken


class EmbeddingManager:
    """Manages text embeddings for the vector store."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_chunk_size: int = 2000):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.max_chunk_size = max_chunk_size
        
        # Initialize tokenizer for text chunking
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        if not text or not text.strip():
            # Return zero embedding for empty text
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Create embedding
        embedding = self.model.encode(cleaned_text, convert_to_numpy=True)
        return embedding.tolist()
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts efficiently."""
        if not texts:
            return []
        
        # Clean all texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Filter out empty texts
        valid_texts = [text for text in cleaned_texts if text.strip()]
        
        if not valid_texts:
            # Return zero embeddings for all texts
            dim = self.model.get_sentence_embedding_dimension()
            return [[0.0] * dim] * len(texts)
        
        # Create embeddings
        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
        
        # Convert to list format
        return [emb.tolist() for emb in embeddings]
    
    def chunk_text(self, text: str, overlap: int = 200) -> List[str]:
        """Split long text into chunks with overlap."""
        if not text or not text.strip():
            return []
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        # If text is short enough, return as single chunk
        if len(tokens) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.max_chunk_size, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start index, accounting for overlap
            start_idx = end_idx - overlap
            
            # Avoid infinite loop
            if start_idx >= end_idx:
                break
        
        return chunks
    
    def create_chunked_embeddings(self, text: str, overlap: int = 200) -> List[Dict[str, Any]]:
        """Create embeddings for text chunks with metadata."""
        chunks = self.chunk_text(text, overlap)
        
        if not chunks:
            return []
        
        embeddings = self.create_embeddings_batch(chunks)
        
        # Combine chunks with embeddings and metadata
        chunked_embeddings = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunked_embeddings.append({
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding,
                "token_count": len(self.tokenizer.encode(chunk)),
                "start_char": text.find(chunk) if chunk in text else 0
            })
        
        return chunked_embeddings
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_most_similar(self, query_embedding: List[float], candidate_embeddings: List[List[float]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find the most similar embeddings to the query."""
        if not query_embedding or not candidate_embeddings:
            return []
        
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate_embedding)
            similarities.append({
                "index": i,
                "similarity": similarity
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top K results
        return similarities[:top_k]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for embedding."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very long repeated characters
        text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)
        
        # Trim to reasonable length
        if len(text) > 10000:
            text = text[:10000]
        
        return text.strip()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()
    
    def aggregate_embeddings(self, embeddings: List[List[float]], method: str = "mean") -> List[float]:
        """Aggregate multiple embeddings into a single embedding."""
        if not embeddings:
            return [0.0] * self.get_embedding_dimension()
        
        embeddings_array = np.array(embeddings)
        
        if method == "mean":
            aggregated = np.mean(embeddings_array, axis=0)
        elif method == "max":
            aggregated = np.max(embeddings_array, axis=0)
        elif method == "sum":
            aggregated = np.sum(embeddings_array, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        
        return aggregated.tolist()