# Script auxiliar para usar Ollama sem precisar do PATH
# Uso: .\ollama.ps1 pull llama3.2
#      .\ollama.ps1 list
#      .\ollama.ps1 run llama3.2

$OllamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

if (Test-Path $OllamaPath) {
    & $OllamaPath $args
} else {
    Write-Error "Ollama não encontrado em: $OllamaPath"
    Write-Host "Baixe em: https://ollama.com/download"
}
