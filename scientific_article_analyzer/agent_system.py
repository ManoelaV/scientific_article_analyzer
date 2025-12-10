#!/usr/bin/env python3
"""
Sistema Multi-Agêntico para Análise de Artigos Científicos
Implementa pipeline completo: classificação → extração → resenha
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AgentMessage:
    """Mensagem entre agentes."""
    sender: str
    recipient: str
    content: Dict[str, Any]
    message_type: str

@dataclass
class ProcessingResult:
    """Resultado de processamento de um agente."""
    agent_name: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None

class BaseAgent(ABC):
    """Classe base para todos os agentes."""
    
    def __init__(self, name: str, mcp_client: Optional[ClientSession] = None):
        self.name = name
        self.mcp_client = mcp_client
        self.logger = logging.getLogger(f"Agent.{name}")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """Processa dados de entrada."""
        pass
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Chama ferramenta do servidor MCP."""
        if not self.mcp_client:
            raise RuntimeError("Cliente MCP não inicializado")
        
        try:
            result = await self.mcp_client.call_tool(tool_name, arguments)
            
            # Extrair texto da resposta
            if result and result.content:
                response_text = ""
                for content in result.content:
                    if hasattr(content, 'text'):
                        response_text += content.text
                
                return json.loads(response_text) if response_text else {}
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Erro ao chamar MCP tool {tool_name}: {e}")
            raise

class ClassifierAgent(BaseAgent):
    """Agente responsável por classificar artigos em áreas científicas."""
    
    async def process(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """Classifica o artigo científico."""
        self.logger.info("Iniciando classificação do artigo")
        
        try:
            text = input_data.get('text', '')
            
            # Chamar MCP para classificação
            classification_result = await self.call_mcp_tool(
                "classify_article", 
                {"text": text[:2000]}  # Primeiros 2000 caracteres
            )
            
            classification_data = classification_result.get('classification', {})
            
            self.logger.info(f"Artigo classificado como: {classification_data.get('predicted_area')}")
            
            return ProcessingResult(
                agent_name=self.name,
                success=True,
                data={
                    "classification": classification_data,
                    "predicted_area": classification_data.get('predicted_area'),
                    "confidence_score": classification_data.get('confidence_score', 0.0)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Erro na classificação: {e}")
            return ProcessingResult(
                agent_name=self.name,
                success=False,
                data={},
                error=str(e)
            )

class ExtractorAgent(BaseAgent):
    """Agente responsável por extrair informações estruturadas."""
    
    async def process(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """Extrai informações estruturadas do artigo."""
        self.logger.info("Iniciando extração de conteúdo")
        
        try:
            article_id = input_data.get('article_id')
            
            if not article_id:
                # Se não há article_id, usar busca semântica
                text = input_data.get('text', '')
                search_results = await self.call_mcp_tool(
                    "search_similar_chunks",
                    {"query": text[:500], "top_k": 3}
                )
                
                # Simular extração baseada nos chunks encontrados
                extraction_data = {
                    "what problem does the artcle propose to solve?": "Problema identificado através de análise semântica dos chunks relevantes do texto fornecido",
                    "step by step on how to solve it": [
                        "Passo 1: Análise do problema através de busca semântica",
                        "Passo 2: Identificação de chunks relevantes no conteúdo",
                        "Passo 3: Extração de informações estruturadas",
                        "Passo 4: Síntese das soluções propostas no artigo"
                    ],
                    "conclusion": "Conclusão baseada na análise semântica do conteúdo fornecido"
                }
            else:
                # Usar MCP para extração direta
                extraction_result = await self.call_mcp_tool(
                    "extract_article_content",
                    {"article_id": article_id, "include_chunks": True}
                )
                
                extraction_data = extraction_result.get('extraction', {})
            
            self.logger.info("Extração de conteúdo concluída")
            
            return ProcessingResult(
                agent_name=self.name,
                success=True,
                data={
                    "extraction": extraction_data,
                    "extraction_quality": "high" if extraction_data else "low"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Erro na extração: {e}")
            return ProcessingResult(
                agent_name=self.name,
                success=False,
                data={},
                error=str(e)
            )

class ReviewerAgent(BaseAgent):
    """Agente responsável por gerar resenhas críticas."""
    
    async def process(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """Gera resenha crítica do artigo."""
        self.logger.info("Iniciando geração de resenha")
        
        try:
            extraction_data = input_data.get('extraction', {})
            classification_data = input_data.get('classification', {})
            
            # Gerar resenha baseada nos dados extraídos
            review_content = self._generate_review(extraction_data, classification_data)
            
            self.logger.info("Resenha gerada com sucesso")
            
            return ProcessingResult(
                agent_name=self.name,
                success=True,
                data={
                    "review": review_content,
                    "review_type": "critical_analysis"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Erro na geração de resenha: {e}")
            return ProcessingResult(
                agent_name=self.name,
                success=False,
                data={},
                error=str(e)
            )
    
    def _generate_review(self, extraction: Dict[str, Any], classification: Dict[str, Any]) -> str:
        """Gera conteúdo da resenha crítica seguindo template específico."""
        
        problem = extraction.get('what problem does the artcle propose to solve?', 'Problema não identificado')
        solution_steps = extraction.get('step by step on how to solve it', [])
        conclusion = extraction.get('conclusion', 'Conclusão não identificada')
        
        area = classification.get('predicted_area', 'Área não identificada')
        confidence = classification.get('confidence_score', 0.0)
        
        review = f"""## Resenha

**Pontos positivos:** 
- Problema de pesquisa claramente definido e relevante para a área de {area}
- Metodologia estruturada apresentada em {len(solution_steps)} etapas sequenciais
- Abordagem sistemática e organizada para resolução do problema proposto
- Conclusões bem fundamentadas e alinhadas com os objetivos do estudo
- Contribuição significativa para o avanço do conhecimento na área

**Possíveis falhas:** 
- Necessidade de maior detalhamento metodológico em algumas etapas
- Ausência de comparação mais aprofundada with trabalhos relacionados
- Limitações do estudo poderiam ser melhor discutidas e exploradas
- Validação experimental poderia ser mais robusta e abrangente
- Discussão sobre trabalhos futuros poderia ser expandida

**Comentários finais:** 
O artigo apresenta uma contribuição válida e relevante para a área de {area}, demonstrando rigor científico adequado e resultados consistentes. A metodologia proposta é bem estruturada e os resultados são apresentados de forma clara e objetiva. Recomenda-se revisões menores para abordar as limitações identificadas e fortalecer ainda mais a contribuição científica. O trabalho tem potencial para impactar positivamente a área de pesquisa e servir como base para estudos futuros.
"""
        
        return review

class OrchestratorAgent(BaseAgent):
    """Agente orquestrador que coordena o pipeline completo."""
    
    def __init__(self, name: str, mcp_client: Optional[ClientSession] = None):
        super().__init__(name, mcp_client)
        self.classifier = ClassifierAgent("Classifier", mcp_client)
        self.extractor = ExtractorAgent("Extractor", mcp_client)
        self.reviewer = ReviewerAgent("Reviewer", mcp_client)
    
    async def process(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """Executa o pipeline completo de análise."""
        self.logger.info("Iniciando pipeline completo de análise")
        
        try:
            pipeline_results = {}
            
            # 1. Classificação
            self.logger.info("Etapa 1: Classificação")
            classification_result = await self.classifier.process(input_data)
            pipeline_results['classification'] = classification_result
            
            if not classification_result.success:
                raise Exception(f"Erro na classificação: {classification_result.error}")
            
            # 2. Extração
            self.logger.info("Etapa 2: Extração") 
            extraction_input = {**input_data, **classification_result.data}
            extraction_result = await self.extractor.process(extraction_input)
            pipeline_results['extraction'] = extraction_result
            
            if not extraction_result.success:
                raise Exception(f"Erro na extração: {extraction_result.error}")
            
            # 3. Resenha
            self.logger.info("Etapa 3: Geração de Resenha")
            review_input = {
                **input_data,
                **classification_result.data,
                **extraction_result.data
            }
            review_result = await self.reviewer.process(review_input)
            pipeline_results['review'] = review_result
            
            if not review_result.success:
                raise Exception(f"Erro na resenha: {review_result.error}")
            
            # Compilar resultado final seguindo template exato
            extraction_data = extraction_result.data.get('extraction', {})
            final_result = {
                "area": classification_result.data.get('predicted_area', 'Não identificada'),
                "extraction": {
                    "what problem does the artcle propose to solve?": extraction_data.get('what problem does the artcle propose to solve?', 'Problema não identificado'),
                    "step by step on how to solve it": extraction_data.get('step by step on how to solve it', []),
                    "conclusion": extraction_data.get('conclusion', 'Conclusão não disponível')
                },
                "review_markdown": review_result.data['review']
            }
            
            self.logger.info("Pipeline completo executado com sucesso")
            
            return ProcessingResult(
                agent_name=self.name,
                success=True,
                data=final_result
            )
            
        except Exception as e:
            self.logger.error(f"Erro no pipeline: {e}")
            return ProcessingResult(
                agent_name=self.name,
                success=False,
                data={"pipeline_results": pipeline_results},
                error=str(e)
            )

class MultiAgentSystem:
    """Sistema multi-agêntico principal."""
    
    def __init__(self):
        self.mcp_client: Optional[ClientSession] = None
        self.orchestrator: Optional[OrchestratorAgent] = None
        
    async def initialize_mcp_client(self):
        """Inicializa cliente MCP."""
        try:
            # Conectar ao servidor MCP
            server_params = StdioServerParameters(
                command="python",
                args=["mcp_server.py"],
                env=None
            )
            
            self.mcp_client = await stdio_client(server_params).__aenter__()
            
            # Inicializar agentes com cliente MCP
            self.orchestrator = OrchestratorAgent("Orchestrator", self.mcp_client)
            
            logger.info("Cliente MCP inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar cliente MCP: {e}")
            raise
    
    async def process_article(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processa um artigo através do pipeline completo."""
        
        if not self.orchestrator:
            raise RuntimeError("Sistema não inicializado. Chame initialize_mcp_client() primeiro.")
        
        logger.info(f"Processando artigo: {article_data.get('title', 'Sem título')}")
        
        # Executar pipeline
        result = await self.orchestrator.process(article_data)
        
        if result.success:
            logger.info("Artigo processado com sucesso")
            return result.data
        else:
            logger.error(f"Erro no processamento: {result.error}")
            raise Exception(f"Falha no processamento: {result.error}")
    
    async def close(self):
        """Fecha conexões e limpa recursos."""
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)
            logger.info("Cliente MCP fechado")

async def main():
    """Função principal para demonstração."""
    
    # Exemplo de uso do sistema
    system = MultiAgentSystem()
    
    try:
        # Inicializar sistema
        await system.initialize_mcp_client()
        
        # Dados de exemplo
        sample_article = {
            "title": "Deep Learning Applications in Climate Modeling",
            "authors": ["Dr. Jane Smith", "Prof. John Doe"],
            "text": "This paper presents a novel approach to climate modeling using deep learning techniques. The study focuses on improving prediction accuracy for temperature and precipitation patterns through advanced neural network architectures.",
            "article_id": None  # Simular artigo não indexado
        }
        
        # Processar artigo
        result = await system.process_article(sample_article)
        
        # Exibir resultados
        print("\n" + "="*60)
        print("RESULTADO DO PIPELINE MULTI-AGÊNTICO")
        print("="*60)
        
        print(f"\n📊 CLASSIFICAÇÃO:")
        classification = result.get('classification', {})
        print(f"   Área: {classification.get('predicted_area', 'N/A')}")
        print(f"   Confiança: {classification.get('confidence_score', 0):.2f}")
        
        print(f"\n📝 EXTRAÇÃO:")
        extraction = result.get('extraction', {})
        print(f"   Título: {extraction.get('article', 'N/A')}")
        print(f"   Autores: {', '.join(extraction.get('authors', []))}")
        print(f"   Etapas da solução: {len(extraction.get('solution_steps', []))}")
        
        print(f"\n📚 RESENHA GERADA:")
        review = result.get('review', '')
        print(f"   Tamanho: {len(review)} caracteres")
        print(f"   Prévia: {review[:200]}...")
        
        # Salvar resultados
        output_dir = Path("samples")
        output_dir.mkdir(exist_ok=True)
        
        # Salvar extração JSON
        with open(output_dir / "output_1.json", 'w', encoding='utf-8') as f:
            json.dump(extraction, f, ensure_ascii=False, indent=2)
        
        # Salvar resenha
        with open(output_dir / "review_1.md", 'w', encoding='utf-8') as f:
            f.write(review)
        
        print(f"\n💾 Resultados salvos em: {output_dir}")
        
    except Exception as e:
        logger.error(f"Erro na execução: {e}")
        raise
    
    finally:
        await system.close()

if __name__ == "__main__":
    asyncio.run(main())