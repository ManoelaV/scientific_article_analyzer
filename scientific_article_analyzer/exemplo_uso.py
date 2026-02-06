#!/usr/bin/env python3
"""
Exemplo de uso do Scientific Article Analyzer
Demonstra como usar o sistema para analisar artigos científicos

NOTA: Alguns exemplos requerem chave de API do OpenAI.
Configure a variável de ambiente OPENAI_API_KEY para usar análise completa.
"""

import asyncio
import os
from main import ScientificArticleAnalyzer
from src.models import ScientificCategory

# Verificar se há API key configurada
HAS_API_KEY = bool(os.getenv("OPENAI_API_KEY"))

async def exemplo_basico():
    """Exemplo básico de análise de artigo (requer API key)"""
    
    if not HAS_API_KEY:
        print("=" * 60)
        print("Exemplo 1: Análise Completa (PULADO - requer API key)")
        print("=" * 60)
        print("\n⚠️  Este exemplo requer uma chave de API do OpenAI.")
        print("   Configure: export OPENAI_API_KEY='sua-chave-aqui'")
        print("   Ou crie um arquivo .env com: OPENAI_API_KEY=sua-chave-aqui")
        return
    
    print("=" * 60)
    print("Exemplo 1: Análise de texto sobre Machine Learning")
    print("=" * 60)
    
    # Inicializar o analisador
    analyzer = ScientificArticleAnalyzer()
    await analyzer.initialize()
    
    # Texto de exemplo sobre Machine Learning
    artigo_ml = """
    Deep Learning for Image Classification
    
    This paper presents a comprehensive study on deep learning techniques for image classification.
    We propose a novel convolutional neural network architecture that achieves state-of-the-art
    performance on ImageNet dataset.
    
    Our method uses attention mechanisms and residual connections to improve feature extraction.
    The proposed architecture consists of multiple convolutional layers with batch normalization
    and ReLU activation functions.
    
    Experimental results show that our approach outperforms existing methods by 5% on accuracy
    while maintaining computational efficiency. We demonstrate the effectiveness of our method
    on various benchmark datasets including CIFAR-10, CIFAR-100, and ImageNet.
    
    In conclusion, our proposed deep learning architecture provides significant improvements
    in image classification tasks through the use of attention mechanisms and optimized
    network design.
    """
    
    # Analisar o artigo
    resultado = await analyzer.analyze_article(
        input_data=artigo_ml,
        input_type="text"
    )
    
    # Mostrar resultados
    print("\n📊 RESULTADOS DA ANÁLISE:")
    print("-" * 60)
    print(f"Categoria: {resultado.classification.category.value}")
    print(f"Confiança: {resultado.classification.confidence:.2%}")
    print(f"\nRaciocínio: {resultado.classification.reasoning}")
    
    print("\n📝 INFORMAÇÕES EXTRAÍDAS:")
    print("-" * 60)
    print(f"Problema: {resultado.extracted_info.problem}")
    print(f"\nPassos da solução:")
    for i, passo in enumerate(resultado.extracted_info.solution_steps, 1):
        print(f"  {i}. {passo}")
    print(f"\nConclusão: {resultado.extracted_info.conclusion}")
    
    print("\n⭐ RESENHA CRÍTICA:")
    print("-" * 60)
    print(f"Resumo: {resultado.review.summary}")
    print(f"\nAspectos Positivos:")
    for aspecto in resultado.review.positive_aspects:
        print(f"  ✓ {aspecto}")
    print(f"\nPossíveis Problemas:")
    for problema in resultado.review.potential_issues:
        print(f"  ⚠ {problema}")
    print(f"\nScore Geral: {resultado.review.overall_score:.1f}/10")


async def exemplo_busca_similaridade():
    """Exemplo de busca por artigos similares"""
    
    print("\n" + "=" * 60)
    print("Exemplo 2: Busca por Artigos Similares")
    print("=" * 60)
    
    analyzer = ScientificArticleAnalyzer()
    await analyzer.initialize()
    
    # Buscar artigos similares sobre deep learning
    query = "deep learning neural networks"
    resultados = await analyzer.search_similar_articles(
        query=query,
        category=ScientificCategory.COMPUTER_SCIENCE,
        limit=3
    )
    
    print(f"\n🔍 Busca: '{query}'")
    print(f"📚 {resultados['total_results']} artigos encontrados:")
    print("-" * 60)
    
    for i, artigo in enumerate(resultados['results'], 1):
        print(f"\n{i}. {artigo['tit (requer API key)"""
    
    if not HAS_API_KEY:
        print("\n" + "=" * 60)
        print("Exemplo 3: Análise de Física (PULADO - requer API key)")
        print("=" * 60)
        print("\n⚠️  Este exemplo requer uma chave de API do OpenAI.")
        return]}")
        print(f"   Similaridade: {artigo['similarity']:.3f}")
        print(f"   Categoria: {artigo['category']}")
        if artigo.get('abstract'):
            preview = artigo['abstract'][:150] + "..." if len(artigo['abstract']) > 150 else artigo['abstract']
            print(f"   Resumo: {preview}")


async def exemplo_artigo_fisica():
    """Exemplo com artigo de física"""
    
    print("\n" + "=" * 60)
    print("Exemplo 3: Análise de Artigo de Física")
    print("=" * 60)
    
    analyzer = ScientificArticleAnalyzer()
    await analyzer.initialize()
    
    artigo_fisica = """
    Quantum Entanglement in Superconducting Qubits
    
    This research investigates quantum entanglement phenomena in superconducting qubit systems.
    We develop a theoretical framework for understanding entanglement dynamics in these quantum
    systems and validate our predictions through experimental measurements.
    
    The problem addressed is the decoherence of quantum states in superconducting circuits,
    which limits the fidelity of quantum operations. Our approach involves designing optimized
    pulse sequences that maintain entanglement while minimizing environmental noise.
    
    We demonstrate successful creation and measurement of Bell states with fidelity exceeding
    99%. The experimental setup uses a dilution refrigerator operating at 10 millikelvin to
    minimize thermal noise effects on the quantum system.
    
    Our findings contribute to the development of practical quantum computing architectures
    and provide insights into quantum information processing in solid-state systems.
    """
    
    resultado = await analyzer.analyze_article(
        input_data=artigo_fisica,
        input_type="text"
    )
    
    print(f"\n📊 Categoria detectada: {resultado.classification.category.value}")
    print(f"   Confiança: {resultado.classification.confidence:.2%}")
    print(f"\n📝 Problema: {resultado.extracted_info.problem}")
    print(f"⭐ Score: {resultado.review.overall_score:.1f}/10")


async def exemplo_estatisticas():
    """Mostrar estatísticas do sistema"""
    
    print("\n" + "=" * 60)
    print("Exemplo 4: Estatísticas do Sistema")
    print("=" * 60)
    
    analyzer = ScientificArticleAnalyzer()
    await analyzer.initialize()
    
    stats = await analyzer.get_system_stats()
    
    print("\n📊 ESTATÍSTICAS DO VECTOR STORE:")
    print("-" * 60)
    
    for categoria, info in stats['vector_store'].items():
        if isinstance(info, dict) and 'count' in info:
            print(f"\n{categoria.upper()}:")
            print(f"  Artigos: {info['count']}")
            if 'articles' in info:
                for artigo in info['articles']:
                    print(f"  - {artigo.get('title', 'Sem título')}")


asynif HAS_API_KEY:
        print("\n✅ Chave de API OpenAI detectada - todos os exemplos disponíveis")
    else:
        print("\n⚠️  Executando sem chave de API - exemplos limitados")
        print("   Para análise completa, configure: OPENAI_API_KEY")
    
    try:
        # Executar exemplos
        await exemplo_basico()
        await exemplo_busca_similaridade()
        await exemplo_artigo_fisica()
        await exemplo_estatisticas()
        
        print("\n" + "=" * 70)
        print("✅ Exemplos executados com sucesso!")
        print("=" * 70)
        
        print("\n💡 PRÓXIMOS PASSOS:")
        if not HAS_API_KEY:
            print("\n   📌 CONFIGURAR API KEY (recomendado):")
            print("   1. Obtenha uma chave em: https://platform.openai.com/api-keys")
            print("   2. Configure a variável de ambiente:")
            print("      Windows: set OPENAI_API_KEY=sua-chave-aqui")
            print("      Linux/Mac: export OPENAI_API_KEY=sua-chave-aqui")
            print("   3. Ou crie arquivo .env com: OPENAI_API_KEY=sua-chave-aqui")
            print()
        print("   📚 EXPERIMENTE:")
        print("   - Modifique os textos de exemplo para seus próprios artigos")
        print("   - Experimente analisar PDFs ou URLs de artigos reais")
        print("   -os os exemplos foram executados com sucesso!")
        print("=" * 70)
        
        print("\n💡 PRÓXIMOS PASSOS:")
        print("   1. Modifique os textos de exemplo para seus próprios artigos")
        print("   2. Adicione suas chaves de API (OPENAI_API_KEY) para análises mais avançadas")
        print("   3. Experimente analisar PDFs ou URLs de artigos reais")
        print("   4. Adicione artigos de referência ao vector store")
        print()
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
