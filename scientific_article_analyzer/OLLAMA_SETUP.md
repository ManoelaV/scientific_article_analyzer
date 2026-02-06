# 🦙 Guia de Configuração do Ollama

Este guia mostra como usar o sistema com **Ollama** (IA local e gratuita).

## 📥 Passo 1: Instalar Ollama

### Windows
1. Baixe o instalador: https://ollama.com/download/windows
2. Execute o instalador
3. Ollama iniciará automaticamente na porta 11434

### Verificar instalação
```powershell
ollama --version
```

## 📦 Passo 2: Baixar um Modelo

Escolha um modelo baseado na sua RAM disponível:

```powershell
# Recomendado - Rápido e leve (4GB RAM)
ollama pull llama3.2

# Alternativas
ollama pull llama3.2:1b      # Ultra leve (1GB RAM)
ollama pull mistral          # Bom equilíbrio (4GB RAM)
ollama pull llama3.1:8b      # Mais poderoso (8GB RAM)
ollama pull llama3.1:70b     # Melhor qualidade (64GB RAM)
```

## ⚙️ Passo 3: Configurar o Sistema

O arquivo `.env` já está configurado para Ollama:

```env
OPENAI_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
```

### Trocar de modelo
Edite `OPENAI_MODEL` no arquivo `.env`:
```env
OPENAI_MODEL=mistral           # Para usar Mistral
OPENAI_MODEL=llama3.1:8b       # Para usar Llama 3.1 8B
```

## 🚀 Passo 4: Usar o Sistema

```powershell
python main.py
```

Ou use o exemplo:
```powershell
python exemplo_uso.py
```

## 🧪 Testar Conexão com Ollama

Primeiro verifique se Ollama está rodando:

```powershell
# Verificar se servidor está ativo
curl http://localhost:11434/api/version

# Testar com o modelo
ollama run llama3.2 "Hello, who are you?"
```

## 📝 Exemplo de Uso

```python
from main import ScientificArticleAnalyzer

# O sistema carrega automaticamente do .env
analyzer = ScientificArticleAnalyzer()
await analyzer.initialize()

# Analisar artigo
article_text = """
Title: Neural Networks for Image Classification

Abstract: This paper presents a novel deep learning architecture...
"""

result = await analyzer.analyze_article(article_text, "text")
print(f"Categoria: {result.classification.category}")
print(f"Problema: {result.extracted_info.problem}")
```

## 🔄 Outras Opções de IA

### Groq (API gratuita e rápida)
```env
OPENAI_API_KEY=sua-chave-groq-aqui
OPENAI_API_BASE=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-70b-versatile
```
Obtenha chave grátis: https://console.groq.com/

### LM Studio (Interface Gráfica)
```env
OPENAI_API_KEY=lm-studio
OPENAI_API_BASE=http://localhost:1234/v1
OPENAI_MODEL=llama-3.2
```
Download: https://lmstudio.ai/

### OpenAI (API paga)
```env
OPENAI_API_KEY=sk-proj-...
# Remover OPENAI_API_BASE
OPENAI_MODEL=gpt-4o-mini
```

## ⚡ Dicas de Performance

1. **Para análises rápidas**: Use `llama3.2:1b` (mais rápido, menos preciso)
2. **Equilíbrio**: Use `llama3.2` ou `mistral` (padrão recomendado)
3. **Máxima qualidade**: Use `llama3.1:70b` (requer muito RAM)

## 🐛 Problemas Comuns

### Erro: "Connection refused"
- Verifique se Ollama está rodando: `ollama serve`
- Windows: Ollama inicia automaticamente, mas pode verificar na bandeja do sistema

### Erro: "Model not found"
- Baixe o modelo primeiro: `ollama pull llama3.2`
- Verifique modelos instalados: `ollama list`

### Sistema muito lento
- Use modelo menor: `llama3.2:1b`
- Ou tente Groq (API na nuvem, gratuita e rápida)

## 📊 Comparação de Modelos

| Modelo | RAM Mínima | Velocidade | Qualidade | Recomendado para |
|--------|------------|------------|-----------|------------------|
| llama3.2:1b | 1GB | ⚡⚡⚡ | ⭐⭐ | Testes rápidos |
| llama3.2 | 4GB | ⚡⚡ | ⭐⭐⭐ | Uso geral |
| mistral | 4GB | ⚡⚡ | ⭐⭐⭐ | Análises técnicas |
| llama3.1:8b | 8GB | ⚡ | ⭐⭐⭐⭐ | Análises detalhadas |
| llama3.1:70b | 64GB | 🐌 | ⭐⭐⭐⭐⭐ | Máxima precisão |

## ✅ Verificar Configuração

```python
import os
from dotenv import load_dotenv

load_dotenv()

print(f"API Key: {os.getenv('OPENAI_API_KEY')}")
print(f"API Base: {os.getenv('OPENAI_API_BASE')}")
print(f"Model: {os.getenv('OPENAI_MODEL')}")
```

Deveria mostrar:
```
API Key: ollama
API Base: http://localhost:11434/v1
Model: llama3.2
```
