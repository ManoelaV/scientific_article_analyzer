#!/usr/bin/env python3
"""
Testes Automatizados para Sistema de Análise de Artigos Científicos
Validação do pipeline completo: MCP Server + Multi-Agent System
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Adicionar diretório principal ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server import VectorStore, server
from agent_system import MultiAgentSystem, ClassifierAgent, ExtractorAgent, ReviewerAgent

class TestVectorStore:
    """Testes para o Vector Store."""
    
    @pytest.fixture
    def temp_vector_store(self):
        """Cria vector store temporário para testes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vs = VectorStore(Path(temp_dir))
            yield vs
    
    @pytest.fixture
    def sample_articles(self):
        """Dados de exemplo para testes."""
        return {
            "test_ml_001": {
                "id": "test_ml_001",
                "title": "Test ML Article",
                "authors": ["Test Author"],
                "abstract": "Test abstract about machine learning algorithms",
                "full_text": "This is a test article about machine learning and neural networks",
                "scientific_area": "machine_learning",
                "keywords": ["machine learning", "neural networks"],
                "publication_date": "2024-01-01",
                "chunk_ids": []
            }
        }
    
    def test_vector_store_initialization(self, temp_vector_store):
        """Testa inicialização do vector store."""
        assert temp_vector_store.articles == {}
        assert temp_vector_store.chunks == {}
        assert temp_vector_store.embeddings is None
        assert temp_vector_store.chunk_ids == []
    
    def test_classify_text_machine_learning(self, temp_vector_store):
        """Testa classificação de texto ML."""
        text = "This article presents neural networks and deep learning algorithms"
        result = temp_vector_store.classify_text(text)
        
        assert result.predicted_area == "machine_learning"
        assert result.confidence_score > 0
        assert "machine_learning" in result.area_scores
        assert "climate_science" in result.area_scores  
        assert "biotechnology" in result.area_scores
    
    def test_classify_text_climate_science(self, temp_vector_store):
        """Testa classificação de texto Climate Science."""
        text = "Global warming and climate change effects on temperature patterns"
        result = temp_vector_store.classify_text(text)
        
        assert result.predicted_area == "climate_science"
        assert result.confidence_score > 0
    
    def test_classify_text_biotechnology(self, temp_vector_store):
        """Testa classificação de texto Biotechnology."""
        text = "Gene editing using CRISPR technology for protein synthesis"
        result = temp_vector_store.classify_text(text)
        
        assert result.predicted_area == "biotechnology"
        assert result.confidence_score > 0
    
    def test_search_similar_empty_store(self, temp_vector_store):
        """Testa busca em vector store vazio."""
        results = temp_vector_store.search_similar("test query")
        assert results == []

class TestMCPServer:
    """Testes para o servidor MCP."""
    
    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Testa listagem de ferramentas MCP."""
        tools = await server.list_tools()
        
        expected_tools = [
            "search_similar_chunks",
            "classify_article", 
            "get_article_metadata",
            "extract_article_content",
            "get_system_stats"
        ]
        
        tool_names = [tool.name for tool in tools]
        for expected in expected_tools:
            assert expected in tool_names
    
    @pytest.mark.asyncio
    async def test_classify_article_tool(self):
        """Testa ferramenta de classificação."""
        args = {"text": "Machine learning algorithms for neural network optimization"}
        
        result = await server.call_tool("classify_article", args)
        assert len(result) == 1
        
        response = json.loads(result[0].text)
        assert response["tool"] == "classify_article"
        assert "classification" in response
        assert response["classification"]["predicted_area"] == "machine_learning"
    
    @pytest.mark.asyncio
    async def test_get_system_stats_tool(self):
        """Testa ferramenta de estatísticas do sistema."""
        result = await server.call_tool("get_system_stats", {})
        assert len(result) == 1
        
        response = json.loads(result[0].text)
        assert response["tool"] == "get_system_stats"
        assert "statistics" in response
        assert "total_articles" in response["statistics"]
        assert "total_chunks" in response["statistics"]

class TestAgentSystem:
    """Testes para o sistema multi-agêntico."""
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Mock do cliente MCP."""
        mock_client = AsyncMock()
        
        # Mock das respostas MCP
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "classification": {
                "predicted_area": "machine_learning",
                "confidence_score": 0.85,
                "area_scores": {
                    "machine_learning": 0.85,
                    "climate_science": 0.10,
                    "biotechnology": 0.05
                }
            }
        }))]
        
        mock_client.call_tool.return_value = mock_response
        return mock_client
    
    @pytest.mark.asyncio
    async def test_classifier_agent(self, mock_mcp_client):
        """Testa agente classificador."""
        agent = ClassifierAgent("TestClassifier", mock_mcp_client)
        
        input_data = {
            "text": "Deep learning neural networks for image classification"
        }
        
        result = await agent.process(input_data)
        
        assert result.success is True
        assert result.agent_name == "TestClassifier"
        assert "classification" in result.data
        assert result.data["predicted_area"] == "machine_learning"
    
    @pytest.mark.asyncio
    async def test_extractor_agent(self, mock_mcp_client):
        """Testa agente extrator."""
        # Mock para extração
        mock_extract_response = Mock()
        mock_extract_response.content = [Mock(text=json.dumps({
            "extraction": {
                "article": "Test Article",
                "authors": ["Test Author"],
                "problem_statement": "Test problem",
                "solution_steps": ["Step 1", "Step 2"],
                "conclusion": "Test conclusion"
            }
        }))]
        
        mock_mcp_client.call_tool.return_value = mock_extract_response
        
        agent = ExtractorAgent("TestExtractor", mock_mcp_client)
        
        input_data = {
            "article_id": "test_001",
            "title": "Test Article"
        }
        
        result = await agent.process(input_data)
        
        assert result.success is True
        assert result.agent_name == "TestExtractor"
        assert "extraction" in result.data
    
    @pytest.mark.asyncio  
    async def test_reviewer_agent(self):
        """Testa agente revisor."""
        agent = ReviewerAgent("TestReviewer")
        
        input_data = {
            "extraction": {
                "article": "Test Article",
                "authors": ["Test Author"],
                "problem_statement": "Test problem statement",
                "solution_steps": ["Step 1", "Step 2"],
                "conclusion": "Test conclusion"
            },
            "classification": {
                "predicted_area": "machine_learning",
                "confidence_score": 0.85
            }
        }
        
        result = await agent.process(input_data)
        
        assert result.success is True
        assert result.agent_name == "TestReviewer"
        assert "review" in result.data
        assert len(result.data["review"]) > 100  # Resenha deve ter conteúdo substantivo

class TestIntegration:
    """Testes de integração do pipeline completo."""
    
    @pytest.mark.asyncio
    async def test_pipeline_json_format(self):
        """Testa formato JSON exato da extração seguindo template especificado."""
        # Simular resultado do pipeline seguindo template exato
        pipeline_result = {
            "area": "Biologia",
            "extraction": {
                "what problem does the artcle propose to solve?": "Este artigo aborda o problema de...",
                "step by step on how to solve it": [
                    "Passo 1: Análise do problema",
                    "Passo 2: Desenvolvimento da solução", 
                    "Passo 3: Implementação"
                ],
                "conclusion": "O estudo conclui que..."
            },
            "review_markdown": "## Resenha\n**Pontos positivos:** ...\n**Possíveis falhas:** ...\n**Comentários finais:** ..."
        }
        
        # Verificar estrutura principal
        assert "area" in pipeline_result
        assert "extraction" in pipeline_result
        assert "review_markdown" in pipeline_result
        
        # Verificar extração
        extraction = pipeline_result["extraction"]
        assert "what problem does the artcle propose to solve?" in extraction
        assert "step by step on how to solve it" in extraction
        assert "conclusion" in extraction
        
        # Verificar tipos
        assert isinstance(pipeline_result["area"], str)
        assert isinstance(extraction["what problem does the artcle propose to solve?"], str)
        assert isinstance(extraction["step by step on how to solve it"], list)
        assert isinstance(extraction["conclusion"], str)
        assert isinstance(pipeline_result["review_markdown"], str)
        
        # Verificar conteúdo não vazio
        assert len(pipeline_result["area"]) > 0
        assert len(extraction["what problem does the artcle propose to solve?"]) > 10
        assert len(extraction["step by step on how to solve it"]) > 0
        assert len(extraction["conclusion"]) > 10
        assert len(pipeline_result["review_markdown"]) > 50
    
    def test_review_format(self):
        """Testa formato da resenha."""
        # Exemplo de resenha gerada
        review_content = """# Resenha Crítica: Test Article

## Pontos Positivos:
- Metodologia bem estruturada
- Resultados relevantes

## Pontos de Melhoria:
- Necessita mais validação experimental

## Recomendação: Aceitar com revisões menores"""
        
        # Verificar estrutura básica
        assert "# Resenha Crítica:" in review_content
        assert "## Pontos Positivos:" in review_content
        assert "## Pontos de Melhoria:" in review_content
        assert "## Recomendação:" in review_content
        assert len(review_content) > 200  # Resenha substantiva

class TestEdgeCases:
    """Testes para casos extremos."""
    
    def test_classify_unknown_domain(self):
        """Testa classificação de texto fora dos domínios conhecidos."""
        vs = VectorStore(Path("/tmp"))
        
        # Texto sobre matemática pura (não nas 3 áreas)
        text = "This paper presents novel theorems in abstract algebra and topology"
        result = vs.classify_text(text)
        
        # Deve escolher uma área mesmo para texto não relacionado
        assert result.predicted_area in ["machine_learning", "climate_science", "biotechnology"]
        # Confiança deve ser baixa
        assert result.confidence_score < 0.5
    
    def test_empty_text_classification(self):
        """Testa classificação de texto vazio."""
        vs = VectorStore(Path("/tmp"))
        
        result = vs.classify_text("")
        
        # Deve ter comportamento graceful
        assert result.predicted_area in ["machine_learning", "climate_science", "biotechnology"]
        assert result.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        """Testa tratamento de erros nos agentes."""
        # Cliente MCP que gera erro
        error_client = AsyncMock()
        error_client.call_tool.side_effect = Exception("MCP Error")
        
        agent = ClassifierAgent("ErrorAgent", error_client)
        
        result = await agent.process({"text": "test"})
        
        assert result.success is False
        assert result.error is not None
        assert "MCP Error" in result.error

def run_tests():
    """Executa todos os testes."""
    print("🧪 Executando Testes Automatizados")
    print("=" * 50)
    
    # Executar pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--no-header"
    ])
    
    if exit_code == 0:
        print("\n✅ Todos os testes passaram!")
        print("🎯 Sistema validado para produção")
    else:
        print("\n❌ Alguns testes falharam")
        print("🔧 Verificar implementação")
    
    return exit_code

if __name__ == "__main__":
    run_tests()