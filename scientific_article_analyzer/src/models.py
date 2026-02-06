"""Data models for the Scientific Article Analysis System"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ScientificCategory(str, Enum):
    """Categories for scientific article classification"""
    COMPUTER_SCIENCE = "computer_science"
    PHYSICS = "physics"
    BIOLOGY = "biology"
    UNKNOWN = "unknown"


@dataclass
class ArticleContent:
    """Represents the content of a scientific article"""
    text: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationResult:
    """Result of article classification"""
    category: ScientificCategory
    confidence: float
    reasoning: Optional[str] = None
    keywords_found: List[str] = field(default_factory=list)


@dataclass
class ExtractedInformation:
    """Information extracted from article in standardized format"""
    problem: str  # "what problem does the article propose to solve?"
    solution_steps: List[str]  # "step by step on how to solve it"
    conclusion: str
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class CriticalReview:
    """Critical review of the article"""
    summary: str
    positive_aspects: List[str]
    potential_issues: List[str]
    overall_score: float  # 1-10
    detailed_scores: Optional[Dict[str, float]] = None
    recommendations: Optional[List[str]] = None


@dataclass
class AnalysisResult:
    """Complete analysis result for an article"""
    classification: ClassificationResult
    extracted_info: ExtractedInformation
    review: CriticalReview
    article_content: ArticleContent
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VectorStoreEntry:
    """Entry in the vector store for reference articles"""
    article_id: str
    category: ScientificCategory
    title: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilaritySearchResult:
    """Result from a similarity search in the vector store"""
    entry: VectorStoreEntry
    similarity_score: float
    rank: int = 0
