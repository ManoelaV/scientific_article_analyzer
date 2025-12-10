#!/usr/bin/env python3
"""
Servidor MCP para Análise de Artigos Científicos
Implementa as ferramentas necessárias para classificação, extração e busca semântica
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field

from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos de dados
class ArticleMetadata(BaseModel):
    """Metadados de um artigo científico."""
    id: str
    title: str
    authors: List[str]
    abstract: str
    full_text: str
    scientific_area: str
    keywords: List[str]
    publication_date: str
    chunk_ids: List[str] = Field(default_factory=list)

class SearchResult(BaseModel):
    """Resultado de busca semântica."""
    chunk_id: str
    content: str
    similarity_score: float
    article_id: str
    page_number: int

class ClassificationResult(BaseModel):
    """Resultado de classificação."""
    predicted_area: str
    confidence_score: float
    area_scores: Dict[str, float]

class ExtractionResult(BaseModel):
    """Resultado de extração - Template exato conforme especificação."""
    what_problem_does_the_artcle_propose_to_solve: str = Field(..., alias="what problem does the artcle propose to solve?", description="Problema que o artigo propõe resolver")
    step_by_step_on_how_to_solve_it: List[str] = Field(..., alias="step by step on how to solve it", description="Passos para resolver")
    conclusion: str = Field(..., description="Conclusão do artigo")
    
    class Config:
        allow_population_by_field_name = True
        populate_by_name = True

class VectorStore:
    """Vector store para armazenamento e busca de embeddings."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.articles: Dict[str, ArticleMetadata] = {}
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_ids: List[str] = []
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        
        # Carregar dados se existirem
        self._load_data()
    
    def _load_data(self):
        """Carrega dados do vector store."""
        articles_file = self.data_dir / "articles.json"
        chunks_file = self.data_dir / "chunks.json" 
        embeddings_file = self.data_dir / "embeddings.npy"
        
        if articles_file.exists():
            with open(articles_file, 'r', encoding='utf-8') as f:
                articles_data = json.load(f)
                self.articles = {
                    aid: ArticleMetadata(**data) 
                    for aid, data in articles_data.items()
                }
        
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
                self.chunk_ids = list(self.chunks.keys())
        
        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)
            
        logger.info(f"Carregados {len(self.articles)} artigos e {len(self.chunks)} chunks")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Busca chunks similares à query."""
        if not self.embeddings.any() or not self.chunk_ids:
            return []
        
        # Gerar embedding da query
        query_embedding = self.encoder.encode([query])
        
        # Calcular similaridades
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Obter top-k resultados
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk_id = self.chunk_ids[idx]
            chunk_data = self.chunks[chunk_id]
            
            results.append(SearchResult(
                chunk_id=chunk_id,
                content=chunk_data['content'],
                similarity_score=float(similarities[idx]),
                article_id=chunk_data['article_id'],
                page_number=chunk_data.get('page_number', 1)
            ))
        
        return results
    
    def classify_text(self, text: str) -> ClassificationResult:
        """Classifica texto em uma das áreas científicas."""
        # Áreas científicas suportadas
        areas = ['machine_learning', 'climate_science', 'biotechnology']
        
        # Keywords por área
        area_keywords = {
            'machine_learning': ['neural', 'learning', 'algorithm', 'model', 'ai', 'deep', 'training'],
            'climate_science': ['climate', 'temperature', 'carbon', 'emission', 'warming', 'environment'],
            'biotechnology': ['gene', 'protein', 'dna', 'bio', 'cell', 'molecular', 'genetic']
        }
        
        # Calcular scores baseado em keywords
        text_lower = text.lower()
        area_scores = {}
        
        for area, keywords in area_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            area_scores[area] = score / len(keywords)  # Normalizar
        
        # Determinar área predita
        predicted_area = max(area_scores, key=area_scores.get)
        confidence = area_scores[predicted_area]
        
        return ClassificationResult(
            predicted_area=predicted_area,
            confidence_score=confidence,
            area_scores=area_scores
        )

# Inicializar servidor MCP
server = Server("scientific-article-analyzer")

# Inicializar vector store
DATA_DIR = Path(__file__).parent / "vector_store"
DATA_DIR.mkdir(exist_ok=True)
vector_store = VectorStore(DATA_DIR)

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """Lista as ferramentas disponíveis do MCP."""
    return [
        Tool(
            name="search_similar_chunks",
            description="Busca chunks similares no vector store",
            inputSchema={
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "Query para busca semântica"},
                    "top_k": {"type": "integer", "description": "Número de resultados (padrão: 5)", "default": 5}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="classify_article",
            description="Classifica artigo em área científica", 
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Texto do artigo para classificação"}
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="get_article_metadata",
            description="Obtém metadados de um artigo",
            inputSchema={
                "type": "object",
                "properties": {
                    "article_id": {"type": "string", "description": "ID do artigo"}
                },
                "required": ["article_id"]
            }
        ),
        Tool(
            name="extract_article_content",
            description="Extrai conteúdo estruturado de um artigo",
            inputSchema={
                "type": "object", 
                "properties": {
                    "article_id": {"type": "string", "description": "ID do artigo"},
                    "include_chunks": {"type": "boolean", "description": "Incluir chunks relacionados", "default": True}
                },
                "required": ["article_id"]
            }
        ),
        Tool(
            name="get_system_stats",
            description="Obtém estatísticas do sistema",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> Sequence[types.TextContent]:
    """Processa chamadas de ferramentas."""
    
    try:
        if name == "search_similar_chunks":
            query = arguments["query"]
            top_k = arguments.get("top_k", 5)
            
            results = vector_store.search_similar(query, top_k)
            
            response = {
                "tool": "search_similar_chunks",
                "query": query,
                "results_count": len(results),
                "results": [result.dict() for result in results]
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]
        
        elif name == "classify_article":
            text = arguments["text"]
            
            result = vector_store.classify_text(text)
            
            response = {
                "tool": "classify_article", 
                "classification": result.dict()
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]
        
        elif name == "get_article_metadata":
            article_id = arguments["article_id"]
            
            if article_id not in vector_store.articles:
                response = {"error": f"Artigo {article_id} não encontrado"}
            else:
                article = vector_store.articles[article_id]
                response = {
                    "tool": "get_article_metadata",
                    "article_id": article_id,
                    "metadata": article.dict()
                }
            
            return [types.TextContent(
                type="text", 
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]
        
        elif name == "extract_article_content":
            article_id = arguments["article_id"] 
            include_chunks = arguments.get("include_chunks", True)
            
            if article_id not in vector_store.articles:
                response = {"error": f"Artigo {article_id} não encontrado"}
            else:
                article = vector_store.articles[article_id]
                
                # Extrair conteúdo estruturado (simulado)
                extraction = ExtractionResult(
                    **{
                        "what problem does the artcle propose to solve?": f"Análise do problema apresentado em {article.title}",
                        "step by step on how to solve it": [
                            "Passo 1: Identificação e definição do problema de pesquisa",
                            "Passo 2: Revisão sistemática da literatura existente", 
                            "Passo 3: Desenvolvimento da metodologia proposta",
                            "Passo 4: Implementação e execução dos experimentos",
                            "Passo 5: Análise e interpretação dos resultados obtidos"
                        ],
                        "conclusion": f"Conclusões baseadas na análise detalhada de {article.title}"
                    }
                )
                
                response = {
                    "tool": "extract_article_content",
                    "article_id": article_id,
                    "extraction": extraction.dict()
                }
                
                if include_chunks:
                    related_chunks = [
                        chunk_data for chunk_id, chunk_data in vector_store.chunks.items()
                        if chunk_data.get('article_id') == article_id
                    ]
                    response["related_chunks"] = related_chunks[:5]  # Top 5
            
            return [types.TextContent(
                type="text",
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]
        
        elif name == "get_system_stats":
            stats = {
                "tool": "get_system_stats",
                "statistics": {
                    "total_articles": len(vector_store.articles),
                    "total_chunks": len(vector_store.chunks),
                    "areas": list(set(article.scientific_area for article in vector_store.articles.values())),
                    "vector_store_size": vector_store.embeddings.shape if vector_store.embeddings is not None else None
                }
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(stats, ensure_ascii=False, indent=2)  
            )]
        
        else:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Ferramenta {name} não reconhecida"})
            )]
    
    except Exception as e:
        logger.error(f"Erro na ferramenta {name}: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

async def main():
    """Executa o servidor MCP."""
    # Executar servidor MCP via stdio
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, 
            write_stream,
            InitializationOptions(
                server_name="scientific-article-analyzer",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())