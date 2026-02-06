# Sistema de Análise de Artigos Científicos

## 🔬 Visão Geral

Sistema completo para análise automatizada de artigos científicos usando MCP (Model Context Protocol), vector store e multi-agentes para classificação, extração e geração de resenhas. Implementa formato de saída padronizado e testes abrangentes incluindo edge cases.

## 🚀 Quick Start

### Windows (PowerShell)
```powershell
# 1. Configurar ambiente
.\run.ps1 setup

# 2. (Opcional) Usar Ollama local
# Instale: https://ollama.com/download
ollama pull llama3.2

# 3. Indexar vector store 
.\run.ps1 index

# 4. Executar testes
.\run.ps1 test1    # Teste com arquivo local
.\run.ps1 test2    # Teste com URL simulada  
.\run.ps1 test3    # Edge case (artigo fora das 3 áreas)
```

### Linux/macOS (Make)
```bash
# 1. Configurar ambiente
make setup

# 2. (Opcional) Usar Ollama local
# Instale: https://ollama.com/download
ollama pull llama3.2

# 3. Indexar vector store
make index

# 4. Executar testes
make test1    # Teste com arquivo local
make test2    # Teste com URL simulada
make test3    # Edge case (artigo fora das 3 áreas)
```

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Funcionalidades](#-funcionalidades)
- [Arquitetura do Sistema](#-arquitetura-do-sistema)
- [Instalação](#-instalação)
- [Configuração](#-configuração)
- [Uso](#-uso)
- [API e Integração](#-api-e-integração)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Exemplos](#-exemplos)
- [Testes](#-testes)
- [Contribuição](#-contribuição)

## 🎯 Visão Geral

O **Scientific Article Analysis System** é uma solução completa que automatiza a análise de artigos científicos através de:

- **Classificação Inteligente**: Categoriza artigos em 3 áreas científicas (Ciência da Computação, Física, Biologia)
- **Extração Estruturada**: Extrai informações em formato JSON específico
- **Resenha Crítica**: Gera análises críticas com aspectos positivos e possíveis falhas
- **Vector Store**: Mantém base de conhecimento com 9 artigos de referência
- **Servidor MCP**: Exporta funcionalidades via Model Context Protocol

## ✨ Funcionalidades

### 🔍 Processamento Multi-formato
- **PDF**: Extração de texto de arquivos PDF
- **URL**: Processamento de artigos online (arXiv, PubMed, etc.)
- **Texto**: Análise direta de conteúdo textual

### 🎯 Classificação Avançada
- Análise por palavras-chave específicas de cada área
- Similaridade semântica com artigos de referência
- Validação por modelos de linguagem (LLM)
- Sistema de confiança com scores

### 📊 Extração Estruturada
Formato JSON padronizado:
```json
{
  "what problem does the article propose to solve?": "...",
  "step by step on how to solve it": ["passo 1", "passo 2", "passo 3"],
  "conclusion": "..."
}
```

### 📝 Resenha Crítica
- Resumo executivo do artigo
- Aspectos positivos identificados
- Possíveis problemas e limitações
- Score geral (1-10)
- Critérios específicos por área científica

### 🗄️ Vector Store
- ChromaDB para armazenamento persistente
- Embeddings com sentence-transformers
- Busca por similaridade semântica
- 9 artigos de referência (3 por categoria)

### 🔌 Servidor MCP
- Interface padronizada via Model Context Protocol
- 7 ferramentas disponíveis
- Integração com sistemas externos
- API RESTful para todas as funcionalidades

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────┐
│                 MCP Server                  │
│           (Interface Externa)               │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│            Main Application                 │
│         (Orquestração Central)              │
└─────┬─────┬─────┬─────┬─────────────────────┘
      │     │     │     │
      ▼     ▼     ▼     ▼
┌──────────────────────────────────────────────┐
│  ArticleProcessor │ Classifier │ Extractor   │
│  (PDF/URL/Text)   │ (3 Áreas)  │ (JSON)      │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│         Vector Store (ChromaDB)              │
│       - 9 Artigos de Referência             │
│       - Embeddings Semânticos               │
│       - Busca por Similaridade              │
└──────────────────────────────────────────────┘
```

## 🚀 Instalação

### Pré-requisitos
- Python 3.10+
- pip (gerenciador de pacotes Python)
- Acesso à internet (para APIs e downloads)

### Passos de Instalação

1. **Clone o repositório**:
```bash
git clone <repository-url>
cd scientific_article_analyzer
```

2. **Crie um ambiente virtual**:
```bash
python -m venv venv
```

3. **Ative o ambiente virtual**:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuração

### 1. Configurar Variáveis de Ambiente

Copie o arquivo de exemplo:
```bash
copy .env.example .env
```

Edite o arquivo `.env` com suas chaves de API.

**Opção A — Ollama (local e gratuito)**
```env
OPENAI_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
```

**Opção B — OpenAI (pago)**
```env
OPENAI_API_KEY=sua_chave_openai_aqui
OPENAI_MODEL=gpt-4o-mini
```

**Opção C — Groq (gratuito)**
```env
OPENAI_API_KEY=sua_chave_groq_aqui
OPENAI_API_BASE=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-70b-versatile
```

Opcional:
```env
ANTHROPIC_API_KEY=sua_chave_anthropic_aqui
```

### 2. Verificar Instalação

Execute o teste simples do sistema:
```bash
python simple_test.py
```


## 📖 Uso

### Interface Principal

```python
from main import ScientificArticleAnalyzer

# Inicializar o sistema
analyzer = ScientificArticleAnalyzer()
await analyzer.initialize()

# Analisar um artigo
result = await analyzer.analyze_article(
    "caminho/para/artigo.pdf",  # ou URL ou texto
    input_type="pdf"  # ou "url" ou "text" ou "auto"
)

# Acessar resultados
print(f"Categoria: {result.classification.category.value}")
print(f"Problema: {result.extracted_info.problem}")
print(f"Score da Resenha: {result.review.overall_score}")
```

### Servidor MCP

Inicie o servidor MCP para integração externa:
```bash
python mcp_server\server.py
```

### Aplicação Standalone

Execute a aplicação principal:
```bash
python main.py
```

## 🔌 API e Integração

### Ferramentas MCP Disponíveis

1. **analyze_article**: Análise completa de artigo
2. **classify_article**: Classificação em categorias
3. **extract_article_info**: Extração de informações JSON
4. **generate_article_review**: Geração de resenha crítica
5. **search_similar_articles**: Busca por similaridade
6. **add_reference_article**: Adicionar artigo de referência
7. **get_vector_store_stats**: Estatísticas do sistema

### Exemplo de Integração

```python
# Via MCP Client
import mcp

client = mcp.Client("localhost:3000")

# Analisar artigo
response = await client.call_tool("analyze_article", {
    "input_data": "texto do artigo...",
    "input_type": "text"
})

result = response["result"]
print(f"Categoria: {result['classification']['category']}")
```

## 📁 Estrutura do Projeto

```
scientific_article_analyzer/
├── src/                          # Componentes principais
│   ├── models.py                # Modelos de dados
│   ├── article_processor.py     # Processamento de artigos
│   ├── classifier.py           # Classificação por categorias
│   ├── extractor.py            # Extração de informações
│   └── reviewer.py             # Geração de resenhas
├── vector_store/               # Sistema de vetores
│   ├── embeddings.py          # Geração de embeddings
│   └── store.py               # Armazenamento ChromaDB
├── mcp_server/                # Servidor MCP
│   ├── tools.py              # Implementação das ferramentas
│   └── server.py             # Servidor principal
├── sample_articles/          # Artigos de referência
│   ├── computer_science/     # 3 artigos de CS
│   ├── physics/              # 3 artigos de Física
│   └── biology/              # 3 artigos de Biologia
├── main.py                   # Aplicação principal
├── test_system.py           # Suite de testes
├── requirements.txt         # Dependências
└── README.md               # Esta documentação
```

## 💡 Exemplos

### Análise de Artigo em PDF

```python
analyzer = ScientificArticleAnalyzer()
await analyzer.initialize()

result = await analyzer.analyze_article(
    "papers/deep_learning_cv.pdf", 
    "pdf"
)

print(f"Categoria: {result.classification.category.value}")
print(f"Confiança: {result.classification.confidence:.2f}")
print(f"Problema identificado: {result.extracted_info.problem}")
```

### Busca por Similaridade

```python
search_results = await analyzer.search_similar_articles(
    "redes neurais convolucionais", 
    category=ScientificCategory.COMPUTER_SCIENCE,
    limit=5
)

for article in search_results["results"]:
    print(f"- {article['title']} (similaridade: {article['similarity']:.3f})")
```

### Adicionar Artigo de Referência

```python
result = await analyzer.add_reference_article(
    "https://arxiv.org/abs/2101.00001",
    ScientificCategory.COMPUTER_SCIENCE,
    "url"
)

if result["success"]:
    print("Artigo adicionado com sucesso!")
```

## 🧪 Testes

### Executar Teste Básico

Para verificar que todos os componentes estão funcionando:
```bash
python simple_test.py
```

Este teste verifica:
- ✅ Importação de todos os módulos
- ✅ Geração de embeddings (384 dimensões)
- ✅ Inicialização do vector store
- ✅ Carregamento do classificador
- ✅ Modelos de dados

### Executar Testes Completos (Em desenvolvimento)

```bash
python test_system.py
```

**Nota:** `test_system.py` foi escrito para uma versão anterior da API. Atualmente, 6 de 16 testes passam (testes do sistema de agentes). Os testes restantes precisam ser atualizados para a API atual.

### Testes Específicos

```bash
# Testar apenas classificação
python -c "
import asyncio
from test_system import SystemTester
async def test(): 
    tester = SystemTester()
    await tester.initialize_system()
    await tester.test_classification()
asyncio.run(test())
"
```

### Cobertura de Testes

- ✅ Processamento de artigos (PDF/URL/texto)
- ✅ Classificação em 3 categorias científicas
- ✅ Extração de informações estruturadas
- ✅ Geração de resenhas críticas
- ✅ Funcionalidade do vector store
- ✅ Integração completa do sistema
- ✅ Estatísticas e status do sistema

## 🔧 Solução de Problemas

### Problemas Comuns

1. **Erro de API Key**:
   - Verifique se a chave OpenAI está correta no arquivo `.env`
   - Certifique-se de que a chave tem créditos disponíveis

2. **Erro de Dependências**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Problema com ChromaDB**:
   ```bash
   # Limpar dados do ChromaDB
   rm -rf ./data/vector_store
   ```

4. **Erro de Memória**:
   - Reduza o `MAX_ARTICLE_LENGTH` no arquivo `.env`
   - Use um modelo de embedding menor

### Logs e Depuração

O sistema gera logs detalhados. Para ativar modo debug:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```


## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

