# Script de Execucao - Sistema de Analise Cientifica
# Equivalente ao Makefile para Windows PowerShell

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    [string]$Input,
    [string]$Url
)

$ErrorActionPreference = "Stop"

function Show-Help {
    Write-Host "SISTEMA DE ANALISE DE ARTIGOS CIENTIFICOS" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Comandos disponiveis:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  setup        " -NoNewline -ForegroundColor Green
    Write-Host "- Instalar dependencias e configurar ambiente"
    Write-Host "  index        " -NoNewline -ForegroundColor Green  
    Write-Host "- Executar notebook de indexacao da vector store"
    Write-Host "  mcp          " -NoNewline -ForegroundColor Green
    Write-Host "- Iniciar servidor MCP (background)"
    Write-Host "  agent        " -NoNewline -ForegroundColor Green
    Write-Host "- Executar sistema multi-agente"
    Write-Host "  test1        " -NoNewline -ForegroundColor Green
    Write-Host "- Teste 1: Processar samples/input_article_1.md"
    Write-Host "  test2        " -NoNewline -ForegroundColor Green
    Write-Host "- Teste 2: Processar artigo via URL (simulado)"
    Write-Host "  test3        " -NoNewline -ForegroundColor Green
    Write-Host "- Teste 3: Edge case (artigo fora das 3 areas)"
    Write-Host "  clean        " -NoNewline -ForegroundColor Green
    Write-Host "- Limpar arquivos temporarios"
    Write-Host "  help         " -NoNewline -ForegroundColor Green
    Write-Host "- Mostrar esta ajuda"
    Write-Host ""
    Write-Host "Exemplos:" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 setup" -ForegroundColor Cyan
    Write-Host "  .\run.ps1 test1" -ForegroundColor Cyan
    Write-Host "  .\run.ps1 agent -Input 'samples/input_article_1.md'" -ForegroundColor Cyan
    Write-Host ""
}

function Invoke-Setup {
    Write-Host "CONFIGURANDO AMBIENTE" -ForegroundColor Cyan
    Write-Host "=====================" -ForegroundColor Gray
    
    # Instalar dependencias
    Write-Host "Instalando dependencias Python..." -ForegroundColor Yellow
    $requirementsFile = "requirements_minimal.txt"
    if (!(Test-Path $requirementsFile)) {
        if (Test-Path "requirements_simple.txt") {
            $requirementsFile = "requirements_simple.txt"
        } elseif (Test-Path "requirements.txt") {
            $requirementsFile = "requirements.txt"
        } else {
            throw "Nenhum arquivo de requirements encontrado"
        }
    }
    Write-Host "Usando: $requirementsFile" -ForegroundColor Gray
    pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) { throw "Erro ao instalar dependencias" }
    
    # Criar diretorios
    Write-Host "Criando diretorios necessarios..." -ForegroundColor Yellow
    @("out", "logs", ".vector_store") | ForEach-Object {
        if (!(Test-Path $_)) {
            New-Item -ItemType Directory -Path $_ -Force | Out-Null
            Write-Host "  [OK] Criado: $_" -ForegroundColor Green
        } else {
            Write-Host "  [OK] Existe: $_" -ForegroundColor Green
        }
    }
    
    Write-Host "Ambiente configurado com sucesso!" -ForegroundColor Green
}

function Invoke-Index {
    Write-Host "EXECUTANDO INDEXACAO DA VECTOR STORE" -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor Gray
    
    if (!(Test-Path "setup_vector_store.ipynb")) {
        throw "Arquivo setup_vector_store.ipynb nao encontrado"
    }
    
    Write-Host "Executando notebook de indexacao..." -ForegroundColor Yellow
    jupyter nbconvert --execute --to notebook --inplace setup_vector_store.ipynb
    if ($LASTEXITCODE -ne 0) { throw "Erro ao executar notebook de indexacao" }
    
    Write-Host "Vector store indexada com sucesso!" -ForegroundColor Green
}

function Invoke-MCP {
    Write-Host "INICIANDO SERVIDOR MCP" -ForegroundColor Cyan
    Write-Host "======================" -ForegroundColor Gray
    
    Write-Host "Iniciando servidor em background..." -ForegroundColor Yellow
    Write-Host "Para parar: Ctrl+C ou feche o terminal" -ForegroundColor Gray
    
    python mcp_server.py
}

function Invoke-Agent {
    Write-Host "EXECUTANDO SISTEMA MULTI-AGENTE" -ForegroundColor Cyan
    Write-Host "===============================" -ForegroundColor Gray
    
    if ($Input) {
        Write-Host "Processando arquivo: $Input" -ForegroundColor Yellow
        python agent_system.py --input $Input
    } else {
        Write-Host "Executando com configuracao padrao..." -ForegroundColor Yellow
        python agent_system.py
    }
    
    if ($LASTEXITCODE -ne 0) { throw "Erro ao executar sistema de agentes" }
    Write-Host "Sistema executado com sucesso!" -ForegroundColor Green
}

function Invoke-Test1 {
    Write-Host "EXECUTANDO TESTE 1" -ForegroundColor Cyan
    Write-Host "==================" -ForegroundColor Gray
    
    python run_test.py --input "samples/input_article_1.md" --output "out/test1_result.json" --review "out/test1_review.md"
    if ($LASTEXITCODE -ne 0) { throw "Teste 1 falhou" }
    
    Write-Host "Teste 1 concluido! Veja out/test1_result.json" -ForegroundColor Green
}

function Invoke-Test2 {
    Write-Host "EXECUTANDO TESTE 2" -ForegroundColor Cyan
    Write-Host "==================" -ForegroundColor Gray
    
    $testUrl = if ($Url) { $Url } else { "https://example.com/climate-paper" }
    python run_test.py --url $testUrl --output "out/test2_result.json" --review "out/test2_review.md"
    if ($LASTEXITCODE -ne 0) { throw "Teste 2 falhou" }
    
    Write-Host "Teste 2 concluido! Veja out/test2_result.json" -ForegroundColor Green
}

function Invoke-Test3 {
    Write-Host "EXECUTANDO TESTE 3 - EDGE CASE" -ForegroundColor Cyan
    Write-Host "===============================" -ForegroundColor Gray
    
    python run_test.py --edge-case --output "out/test3_result.json" --review "out/test3_review.md"
    if ($LASTEXITCODE -ne 0) { throw "Teste 3 falhou" }
    
    Write-Host "Teste 3 concluido! Veja out/test3_result.json" -ForegroundColor Green
}

function Invoke-Clean {
    Write-Host "LIMPANDO ARQUIVOS TEMPORARIOS" -ForegroundColor Cyan
    Write-Host "=============================" -ForegroundColor Gray
    
    @("out/*", "logs/*", "__pycache__", "*.pyc") | ForEach-Object {
        if (Test-Path $_) {
            Remove-Item $_ -Recurse -Force
            Write-Host "  [OK] Removido: $_" -ForegroundColor Green
        }
    }
    
    Write-Host "Limpeza concluida!" -ForegroundColor Green
}

# Execução principal
try {
    switch ($Command.ToLower()) {
        "setup" { Invoke-Setup }
        "index" { Invoke-Index }
        "mcp" { Invoke-MCP }
        "agent" { Invoke-Agent }
        "test1" { Invoke-Test1 }
        "test2" { Invoke-Test2 }
        "test3" { Invoke-Test3 }
        "clean" { Invoke-Clean }
        "help" { Show-Help }
        default { 
            Write-Host "Comando desconhecido: $Command" -ForegroundColor Red
            Show-Help
            exit 1
        }
    }
} catch {
    Write-Host "ERRO: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}