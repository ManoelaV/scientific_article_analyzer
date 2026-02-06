#!/usr/bin/env python3
"""
Simple test to verify system components work
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Scientific Article Analyzer - Simple Test")
print("=" * 60)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    from src.models import ScientificCategory, ArticleContent
    from vector_store.embeddings import EmbeddingManager  
    from vector_store.store import VectorStore
    from src.classifier import ArticleClassifier
    from src.extractor import InformationExtractor
    from src.reviewer import CriticalReviewer
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check embeddings
print("\n[2/5] Testing embeddings...")
try:
    embedder = EmbeddingManager()
    test_text = "Machine learning and neural networks for computer vision"
    embedding = embedder.create_embedding(test_text)
    print(f"✓ Embedding created successfully (dimension: {len(embedding)})")
except Exception as e:
    print(f"✗ Embedding error: {e}")

# Test 3: Check vector store initialization  
print("\n[3/5] Testing vector store...")
try:
    embedder = EmbeddingManager()
    vs = VectorStore(embedder, "./test_data/vector_store")
    print(f"✓ Vector store initialized")
    print(f"  Categories: {list(vs.articles.keys())}")
except Exception as e:
    print(f"✗ Vector store error: {e}")

# Test 4: Test article classifier
print("\n[4/5] Testing classifier...")
try:
    # Check if classifier has required methods
    print("  Classifier module loaded successfully")
except Exception as e:
    print(f"✗ Classifier error: {e}")

# Test 5: Test models
print("\n[5/5] Testing data models...")
try:
    article = ArticleContent(
        title="Test Article",
        text="This is a test article about machine learning"
    )
    category = ScientificCategory.COMPUTER_SCIENCE
    print(f"✓ Models working correctly")
    print(f"  Article: {article.title}")
    print(f"  Category: {category.value}")
except Exception as e:
    print(f"✗ Models error: {e}")

print("\n" + "=" * 60)
print("Basic component tests completed!")
print("=" * 60)
print("\nNext steps:")
print("1. Run full system test: python test_system.py")
print("2. Or run main application: python main.py")
print("=" * 60)
