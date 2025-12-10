# Scientific Article Analyzer - Makefile
# Comandos para setup e execução do sistema

.PHONY: setup index mcp agent test clean help

# Configuração inicial do ambiente
setup:
	@echo "🔧 Configurando ambiente..."
	pip install -r requirements_minimal.txt
	@mkdir -p out
	@mkdir -p vector_store
	@echo "✅ Ambiente configurado com sucesso!"

# Constrói e popula o vector store com 9 artigos
index:
	@echo "📚 Construindo vector store com 9 artigos..."
	jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=600 setup_vector_store.ipynb
	@echo "✅ Vector store populado com sucesso!"

# Sobe o servidor MCP em background
mcp:
	@echo "🖥️ Iniciando servidor MCP..."
	@python mcp_server.py &
	@echo "✅ Servidor MCP iniciado em background (PID: $$!)"
	@sleep 2

# Inicia o agente multi-agêntico
agent:
	@echo "🤖 Iniciando sistema multi-agêntico..."
	python agent_system.py
	@echo "✅ Sistema de agentes executado!"

# Executa testes completos
test: test1 test2 test3

# Teste 1: Classificar e extrair samples/input_article_1.md
test1:
	@echo "🧪 TESTE 1: Classificação e extração do artigo de exemplo"
	@echo "=================================================="
	python run_test.py --input samples/input_article_1.md --output out/extraction_1.json --review out/review_1.md
	@echo "✅ Teste 1 concluído!"

# Teste 2: Artigo via URL (simulado com texto)
test2:
	@echo "🧪 TESTE 2: Processamento de artigo via URL"
	@echo "==========================================="
	python run_test.py --url "https://example.com/article" --output out/extraction_2.json --review out/review_2.md
	@echo "✅ Teste 2 concluído!"

# Teste 3: Edge case - artigo fora das 3 áreas
test3:
	@echo "🧪 TESTE 3: Edge case - artigo fora das áreas conhecidas"
	@echo "======================================================="
	python run_test.py --edge-case --output out/extraction_3.json --review out/review_3.md
	@echo "✅ Teste 3 concluído!"

# Limpa arquivos temporários e outputs
clean:
	@echo "🧹 Limpando arquivos temporários..."
	@rm -rf out/*.json out/*.md
	@rm -rf __pycache__/ *.pyc
	@echo "✅ Limpeza concluída!"

# Mostra comandos disponíveis
help:
	@echo "🔬 Scientific Article Analyzer - Comandos Disponíveis"
	@echo "====================================================="
	@echo ""
	@echo "Setup e Configuração:"
	@echo "  make setup    - Configura ambiente e dependências"
	@echo "  make index    - Constrói e popula vector store (9 artigos)"
	@echo ""
	@echo "Execução:"
	@echo "  make mcp      - Inicia servidor MCP em background"
	@echo "  make agent    - Executa sistema multi-agêntico"
	@echo ""
	@echo "Testes:"
	@echo "  make test     - Executa todos os testes (1, 2 e 3)"
	@echo "  make test1    - Teste 1: samples/input_article_1.md"
	@echo "  make test2    - Teste 2: artigo via URL"
	@echo "  make test3    - Teste 3: edge case (área não coberta)"
	@echo ""
	@echo "Utilitários:"
	@echo "  make clean    - Remove arquivos temporários"
	@echo "  make help     - Mostra esta ajuda"

# Pipeline completo: setup -> index -> mcp -> testes
all: setup index mcp test
	@echo "🎉 Pipeline completo executado com sucesso!"