"""
Simple Vector Store Implementation
Uses numpy and sklearn for vector operations without external vector databases.
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from src.models import (
    ScientificCategory, 
    ArticleContent, 
    VectorStoreEntry, 
    SimilaritySearchResult
)
from .embeddings import EmbeddingManager


class VectorStore:
    """Simple vector store for scientific articles using numpy and sklearn."""
    
    def __init__(self, embedding_manager: EmbeddingManager, 
                 data_dir: str = "./data/vector_store"):
        """
        Initialize the vector store.
        
        Args:
            embedding_manager: Manager for generating embeddings
            data_dir: Directory to store the vector data
        """
        self.embedding_manager = embedding_manager
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for articles
        self.articles = {}  # category -> list of articles
        self.embeddings = {}  # category -> numpy array of embeddings
        
        # Files to persist data
        self.articles_file = self.data_dir / "articles.json"
        self.embeddings_file = self.data_dir / "embeddings.pkl"
        
        # Initialize storage for each category
        for category in ScientificCategory:
            self.articles[category] = []
            self.embeddings[category] = np.array([]).reshape(0, 384)  # 384 is the embedding dimension
            
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load existing articles and embeddings from disk."""
        try:
            # Load articles
            if self.articles_file.exists():
                with open(self.articles_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for category_name, articles_list in data.items():
                        try:
                            category = ScientificCategory(category_name)
                            self.articles[category] = articles_list
                        except ValueError:
                            continue
            
            # Load embeddings
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    embeddings_data = pickle.load(f)
                    for category_name, embeddings_array in embeddings_data.items():
                        try:
                            category = ScientificCategory(category_name)
                            self.embeddings[category] = np.array(embeddings_array)
                        except ValueError:
                            continue
                            
        except Exception as e:
            print(f"Warning: Could not load existing data: {e}")
    
    def _save_data(self):
        """Save articles and embeddings to disk."""
        try:
            # Save articles
            articles_data = {}
            for category, articles_list in self.articles.items():
                articles_data[category.value] = articles_list
            
            with open(self.articles_file, 'w', encoding='utf-8') as f:
                json.dump(articles_data, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            embeddings_data = {}
            for category, embeddings_array in self.embeddings.items():
                embeddings_data[category.value] = embeddings_array.tolist()
            
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings_data, f)
                
        except Exception as e:
            print(f"Warning: Could not save data: {e}")
    
    async def add_article(self, article: ArticleContent, category: ScientificCategory) -> str:
        """
        Add an article to the vector store.
        
        Args:
            article: Article content to add
            category: Scientific category of the article
            
        Returns:
            str: Article ID
        """
        try:
            # Generate embedding
            embedding = self.embedding_manager.create_embedding(article.text)
            
            # Create article entry
            article_id = f"{category.value}_{len(self.articles[category])}"
            article_data = {
                "id": article_id,
                "title": article.title,
                "abstract": article.abstract or article.text[:500],
                "content": article.text,
                "url": article.source_url or "",
                "category": category.value,
                "added_at": datetime.now().isoformat()
            }
            
            # Add to storage
            self.articles[category].append(article_data)
            
            # Add embedding
            embedding_array = np.array(embedding).reshape(1, -1)
            if self.embeddings[category].size == 0:
                self.embeddings[category] = embedding_array
            else:
                self.embeddings[category] = np.vstack([self.embeddings[category], embedding_array])
            
            # Save to disk
            self._save_data()
            
            return article_id
            
        except Exception as e:
            raise Exception(f"Failed to add article: {e}")
    
    async def search_similar(self, query: str, category: Optional[ScientificCategory] = None,
                           limit: int = 5, min_similarity: float = 0.0) -> List[SimilaritySearchResult]:
        """
        Search for similar articles using vector similarity.
        
        Args:
            query: Search query text
            category: Optional category filter
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similarity search results
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_manager.create_embedding(query)
            query_vector = np.array(query_embedding).reshape(1, -1)
            
            results = []
            
            # Search categories
            categories_to_search = [category] if category else list(ScientificCategory)
            
            for cat in categories_to_search:
                if len(self.articles[cat]) == 0:
                    continue
                
                # Calculate similarities
                similarities = cosine_similarity(query_vector, self.embeddings[cat])[0]
                
                # Create results
                for i, similarity in enumerate(similarities):
                    if similarity >= min_similarity:
                        article = self.articles[cat][i]
                        # Create VectorStoreEntry for the result
                        entry = VectorStoreEntry(
                            id=article["id"],
                            content=article["content"],
                            category=ScientificCategory(article["category"]),
                            embedding=self.embeddings[cat][i].tolist(),
                            metadata={
                                "title": article["title"],
                                "abstract": article["abstract"],
                                **article.get("metadata", {})
                            }
                        )
                        
                        results.append(SimilaritySearchResult(
                            entry=entry,
                            similarity_score=float(similarity),
                            relevance_explanation=f"Similarity: {similarity:.3f}"
                        ))
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x.similarity, reverse=True)
            return results[:limit]
            
        except Exception as e:
            raise Exception(f"Search failed: {e}")
    
    def get_reference_articles(self, category: ScientificCategory) -> List[Dict[str, Any]]:
        """
        Get all reference articles for a specific category.
        
        Args:
            category: Scientific category
            
        Returns:
            List of articles in the category
        """
        return self.articles[category].copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary containing store statistics
        """
        stats = {
            "total_articles": sum(len(articles) for articles in self.articles.values()),
            "categories": {},
            "embedding_dimension": 384,
            "storage_path": str(self.data_dir)
        }
        
        for category in ScientificCategory:
            stats["categories"][category.value] = {
                "article_count": len(self.articles[category]),
                "embedding_count": self.embeddings[category].shape[0] if self.embeddings[category].size > 0 else 0
            }
        
        return stats
    
    async def initialize_with_sample_articles(self):
        """Initialize the vector store with sample reference articles."""
        
        # Check if we already have articles
        total_articles = sum(len(articles) for articles in self.articles.values())
        if total_articles > 0:
            print(f"Vector store already contains {total_articles} articles, skipping initialization")
            return
        
        # Sample articles for each category
        sample_articles = {
            ScientificCategory.COMPUTER_SCIENCE: [
                ArticleContent(
                    title="Deep Learning for Computer Vision: A Comprehensive Survey",
                    abstract="This paper provides a comprehensive survey of deep learning techniques for computer vision applications.",
                    text="""
                    Deep learning has revolutionized computer vision in recent years. This comprehensive survey
                    examines the major deep learning architectures used in computer vision, including convolutional
                    neural networks (CNNs), recurrent neural networks (RNNs), and transformer-based models.
                    
                    The paper covers key applications such as image classification, object detection, semantic
                    segmentation, and image generation. We discuss the evolution from early CNN architectures
                    like LeNet and AlexNet to more recent developments such as ResNet, DenseNet, and Vision
                    Transformers.
                    
                    Key contributions include: 1) A systematic categorization of deep learning approaches,
                    2) Performance comparisons across different architectures, 3) Analysis of computational
                    requirements and efficiency considerations.
                    
                    Our experiments show that transformer-based models achieve state-of-the-art performance
                    on many computer vision benchmarks, while CNN architectures remain competitive for
                    real-time applications due to their computational efficiency.
                    """,
                    source_url="https://example.com/dl-cv-survey"
                ),
                ArticleContent(
                    title="Attention Mechanisms in Neural Networks: A Survey",
                    abstract="A comprehensive review of attention mechanisms and their applications in various neural network architectures.",
                    text="""
                    Attention mechanisms have become a fundamental component of modern neural networks,
                    enabling models to focus on relevant parts of input data. This survey provides
                    a comprehensive overview of attention mechanisms across different domains.
                    
                    We categorize attention mechanisms into several types: self-attention, cross-attention,
                    multi-head attention, and sparse attention. The paper traces the evolution from
                    early attention models in machine translation to the transformer architecture
                    that relies entirely on attention mechanisms.
                    
                    Key applications covered include natural language processing, computer vision,
                    speech recognition, and multimodal learning. We analyze the computational
                    complexity and memory requirements of different attention variants.
                    
                    The survey concludes with emerging trends such as efficient attention mechanisms,
                    attention visualization techniques, and the integration of attention with other
                    architectural components for improved performance and interpretability.
                    """,
                    source_url="https://example.com/attention-survey"
                ),
                ArticleContent(
                    title="Reinforcement Learning in Robotics: Recent Advances and Applications",
                    abstract="This paper reviews recent advances in applying reinforcement learning to robotics applications.",
                    text="""
                    Reinforcement learning (RL) has shown remarkable success in robotics applications,
                    enabling robots to learn complex behaviors through interaction with their environment.
                    This paper reviews recent advances in RL for robotics.
                    
                    We cover key RL algorithms adapted for robotics, including policy gradient methods,
                    actor-critic algorithms, and model-based approaches. The paper discusses the
                    challenges of applying RL to real-world robotic systems, including sample
                    efficiency, safety constraints, and sim-to-real transfer.
                    
                    Major applications reviewed include robotic manipulation, locomotion, navigation,
                    and human-robot interaction. We analyze successful case studies such as robotic
                    grasping, bipedal walking, and autonomous drone control.
                    
                    The paper concludes with future directions including meta-learning for robotics,
                    multi-agent robotic systems, and the integration of RL with other AI techniques
                    such as computer vision and natural language processing.
                    """,
                    source_url="https://example.com/rl-robotics"
                )
            ],
            
            ScientificCategory.PHYSICS: [
                ArticleContent(
                    title="Quantum Entanglement in Many-Body Systems: Theory and Applications",
                    abstract="A comprehensive study of quantum entanglement phenomena in many-body quantum systems.",
                    text="""
                    Quantum entanglement is a fundamental phenomenon in quantum mechanics that plays
                    a crucial role in many-body quantum systems. This paper provides a comprehensive
                    theoretical framework for understanding entanglement in complex quantum systems.
                    
                    We develop mathematical tools for characterizing entanglement using entropy measures,
                    entanglement spectrum analysis, and topological indicators. The paper covers both
                    equilibrium and non-equilibrium many-body systems, including quantum spin chains,
                    ultracold atomic gases, and condensed matter systems.
                    
                    Key results include: 1) Universal scaling laws for entanglement entropy,
                    2) Classification of quantum phase transitions based on entanglement properties,
                    3) Novel protocols for entanglement detection in experimental systems.
                    
                    Applications to quantum information processing, quantum simulation, and quantum
                    metrology are discussed. The paper concludes with open problems and future
                    research directions in many-body entanglement theory.
                    """,
                    source_url="https://example.com/quantum-entanglement"
                ),
                ArticleContent(
                    title="Superconductivity in Two-Dimensional Materials: Electronic Properties and Applications",
                    abstract="Investigation of superconducting properties in two-dimensional materials and their potential applications.",
                    text="""
                    Two-dimensional materials have emerged as a new platform for studying unconventional
                    superconductivity. This paper investigates the electronic properties of 2D
                    superconductors and their potential technological applications.
                    
                    We examine various 2D superconducting materials including transition metal
                    dichalcogenides, graphene-based systems, and van der Waals heterostructures.
                    The paper discusses the role of reduced dimensionality on superconducting
                    properties and the emergence of novel pairing mechanisms.
                    
                    Experimental techniques for probing 2D superconductivity are reviewed,
                    including transport measurements, scanning tunneling spectroscopy, and
                    angle-resolved photoemission spectroscopy. We present results on critical
                    temperatures, coherence lengths, and gap symmetries.
                    
                    Applications in quantum electronics, superconducting qubits, and energy
                    storage are explored. The paper concludes with challenges and opportunities
                    for engineering superconducting devices based on 2D materials.
                    """,
                    source_url="https://example.com/2d-superconductors"
                ),
                ArticleContent(
                    title="Dark Matter Detection: Current Status and Future Prospects",
                    abstract="Review of current dark matter detection methods and analysis of future experimental prospects.",
                    text="""
                    Dark matter constitutes approximately 27% of the universe's mass-energy content,
                    yet its nature remains one of the greatest mysteries in physics. This paper
                    reviews current detection methods and analyzes future experimental prospects.
                    
                    We categorize detection approaches into three main types: direct detection
                    through nuclear recoils, indirect detection via cosmic rays, and collider
                    searches for dark matter candidates. The paper discusses the theoretical
                    motivations for various dark matter models including WIMPs, axions, and
                    sterile neutrinos.
                    
                    Current experimental results from underground laboratories, space-based
                    telescopes, and particle accelerators are analyzed. We examine the
                    complementarity of different detection strategies and their sensitivity
                    to various dark matter models.
                    
                    Future experimental programs including next-generation direct detection
                    experiments, improved cosmic ray observatories, and high-luminosity
                    colliders are discussed. The paper concludes with theoretical developments
                    that may guide future search strategies.
                    """,
                    source_url="https://example.com/dark-matter"
                )
            ],
            
            ScientificCategory.BIOLOGY: [
                ArticleContent(
                    title="CRISPR-Cas9 Gene Editing: Mechanisms, Applications, and Therapeutic Potential",
                    abstract="Comprehensive review of CRISPR-Cas9 gene editing technology and its applications in medicine and research.",
                    text="""
                    CRISPR-Cas9 has revolutionized gene editing by providing a precise, efficient,
                    and programmable system for genome modification. This comprehensive review
                    examines the molecular mechanisms, current applications, and therapeutic
                    potential of CRISPR-Cas9 technology.
                    
                    We describe the structure and function of the Cas9 protein, guide RNA design
                    principles, and the DNA repair mechanisms that enable precise editing. The
                    paper covers various CRISPR applications including gene knockout, knock-in,
                    transcriptional regulation, and epigenome editing.
                    
                    Clinical applications are discussed, including treatments for genetic disorders,
                    cancer therapy, and infectious diseases. We analyze successful clinical trials
                    and examine the challenges of in vivo delivery and off-target effects.
                    
                    The review concludes with emerging CRISPR technologies such as base editing,
                    prime editing, and CRISPR 2.0 systems. Ethical considerations and regulatory
                    frameworks for clinical applications are also addressed.
                    """,
                    source_url="https://example.com/crispr-review"
                ),
                ArticleContent(
                    title="Single-Cell RNA Sequencing: Technologies, Analysis Methods, and Biological Insights",
                    abstract="Overview of single-cell RNA sequencing technologies and their applications in understanding cellular heterogeneity.",
                    text="""
                    Single-cell RNA sequencing (scRNA-seq) has transformed our understanding of
                    cellular heterogeneity and gene expression dynamics. This paper provides an
                    overview of scRNA-seq technologies, analysis methods, and biological insights.
                    
                    We review the major scRNA-seq platforms including droplet-based methods,
                    plate-based approaches, and microwell technologies. The paper discusses
                    technical considerations such as cell capture efficiency, transcript coverage,
                    and computational requirements for different platforms.
                    
                    Computational analysis workflows are examined, including quality control,
                    normalization, dimensionality reduction, clustering, and trajectory inference.
                    We present best practices for experimental design and data interpretation.
                    
                    Biological applications span development, immunology, neuroscience, and
                    cancer research. Case studies demonstrate how scRNA-seq has revealed new
                    cell types, characterized differentiation processes, and identified
                    disease mechanisms at single-cell resolution.
                    """,
                    source_url="https://example.com/scrna-seq"
                ),
                ArticleContent(
                    title="Microbiome-Host Interactions: Mechanisms and Implications for Human Health",
                    abstract="Investigation of molecular mechanisms underlying microbiome-host interactions and their impact on health and disease.",
                    text="""
                    The human microbiome plays a crucial role in health and disease through
                    complex interactions with host physiology. This paper investigates the
                    molecular mechanisms underlying microbiome-host interactions and their
                    implications for human health.
                    
                    We examine the composition and diversity of the human microbiome across
                    different body sites, with emphasis on the gut microbiome. The paper
                    discusses factors that influence microbiome composition including diet,
                    genetics, antibiotics, and environmental exposures.
                    
                    Molecular mechanisms of host-microbe communication are analyzed, including
                    metabolite signaling, immune system modulation, and epithelial barrier
                    function. We present evidence for microbiome involvement in metabolism,
                    immune development, and neurological function.
                    
                    Disease associations are explored, including inflammatory bowel disease,
                    obesity, diabetes, and mental health disorders. The paper concludes with
                    therapeutic approaches targeting the microbiome, including probiotics,
                    fecal microbiota transplantation, and precision microbiome interventions.
                    """,
                    source_url="https://example.com/microbiome"
                )
            ]
        }
        
        # Add sample articles to the vector store
        for category, articles in sample_articles.items():
            for article in articles:
                try:
                    await self.add_article(article, category)
                    print(f"Added sample article: {article.title}")
                except Exception as e:
                    print(f"Failed to add sample article {article.title}: {e}")
        
        print(f"Initialized vector store with {sum(len(articles) for articles in sample_articles.values())} reference articles")
