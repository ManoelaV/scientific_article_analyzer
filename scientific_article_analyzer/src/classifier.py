"""Article classifier for categorizing scientific articles"""

import logging
from typing import List, Optional
import re

from .models import ArticleContent, ClassificationResult, ScientificCategory

logger = logging.getLogger(__name__)


class ArticleClassifier:
    """Classifies articles into scientific categories"""

    # Heuristic thresholds to reduce false positives
    MIN_KEYWORD_MATCHES = 3
    MIN_CONFIDENCE = 0.45
    MIN_MARGIN = 0.15
    
    # Keywords for each category
    CATEGORY_KEYWORDS = {
        ScientificCategory.COMPUTER_SCIENCE: [
            "algorithm", "computational", "neural network", "machine learning", 
            "artificial intelligence", "data structure", "programming", "software",
            "computer", "cpu", "gpu", "memory", "cache", "processor", "deep learning",
            "convolutional", "recurrent", "transformer", "neural", "training",
            "inference", "model", "dataset", "accuracy", "performance", "optimization",
            "compiler", "operating system", "database", "network", "protocol"
        ],
        ScientificCategory.PHYSICS: [
            "quantum", "particle", "energy", "force", "electromagnetic", "photon",
            "electron", "atom", "molecule", "thermodynamics", "mechanics", "relativity",
            "gravitational", "wave", "field", "mass", "velocity", "acceleration",
            "momentum", "entropy", "plasma", "nuclear", "fusion", "fission",
            "cosmology", "astrophysics", "spacetime", "black hole", "quasar"
        ],
        ScientificCategory.BIOLOGY: [
            "cell", "protein", "dna", "rna", "gene", "genome", "chromosome",
            "enzyme", "metabolism", "organism", "species", "evolution", "mutation",
            "bacteria", "virus", "tissue", "organ", "neural", "brain", "neuron",
            "ecosystem", "population", "biodiversity", "molecular", "cellular",
            "genetic", "phenotype", "genotype", "antibody", "immune", "pathogen"
        ]
    }
    
    def __init__(self, openai_api_key: Optional[str] = None, api_base: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize classifier with optional OpenAI API key"""
        self.openai_api_key = openai_api_key
        self.api_base = api_base
        self.model = model
        self.use_llm = openai_api_key is not None
    
    async def classify(self, article: ArticleContent) -> ClassificationResult:
        """
        Classify article into a scientific category.
        
        Args:
            article: Article content to classify
            
        Returns:
            ClassificationResult with category and confidence
        """
        # Start with keyword-based classification
        keyword_result = self._classify_by_keywords(article.text)
        
        # If we have LLM access, use it for validation
        if self.use_llm:
            try:
                llm_result = await self._classify_with_llm(article.text, keyword_result)
                return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}. Using keyword classification.")
                return keyword_result
        
        return keyword_result
    
    def _classify_by_keywords(self, text: str) -> ClassificationResult:
        """Classify based on keyword matching"""
        text_lower = text.lower()
        
        scores = {}
        keywords_found = {}
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = 0
            found = []
            
            for keyword in keywords:
                # Count occurrences of keyword
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    score += matches
                    found.append(keyword)
            
            scores[category] = score
            keywords_found[category] = found
        
        # Find category with highest score
        if all(s == 0 for s in scores.values()):
            return ClassificationResult(
                category=ScientificCategory.UNKNOWN,
                confidence=0.0,
                reasoning="No category-specific keywords found",
                keywords_found=[]
            )
        
        best_category = max(scores, key=scores.get)
        total_score = sum(scores.values())
        best_score = scores[best_category]
        sorted_scores = sorted(scores.values(), reverse=True)
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
        confidence = best_score / total_score if total_score > 0 else 0.0
        margin = (best_score - second_score) / total_score if total_score > 0 else 0.0

        # Guardrails to avoid false positives
        if best_score < self.MIN_KEYWORD_MATCHES or confidence < self.MIN_CONFIDENCE or margin < self.MIN_MARGIN:
            return ClassificationResult(
                category=ScientificCategory.UNKNOWN,
                confidence=confidence,
                reasoning=(
                    f"Low evidence: matches={best_score}, confidence={confidence:.2f}, "
                    f"margin={margin:.2f}"
                ),
                keywords_found=keywords_found[best_category][:10]
            )
        
        return ClassificationResult(
            category=best_category,
            confidence=confidence,
            reasoning=f"Keyword analysis: {scores[best_category]} matches",
            keywords_found=keywords_found[best_category][:10]  # Top 10 keywords
        )
    
    async def _classify_with_llm(self, text: str, keyword_result: ClassificationResult) -> ClassificationResult:
        """Use LLM for more accurate classification"""
        try:
            import openai
            
            # Suporte para APIs customizadas (Ollama, LM Studio, etc.)
            client_kwargs = {"api_key": self.openai_api_key}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            
            client = openai.OpenAI(**client_kwargs)
            
            # Truncate text for API
            text_sample = text[:3000] if len(text) > 3000 else text
            
            prompt = f"""Classify the following scientific article excerpt into ONE of these categories:
- computer_science
- physics
- biology

The keyword analysis suggests: {keyword_result.category.value} (confidence: {keyword_result.confidence:.2f})

Article excerpt:
{text_sample}

Respond with ONLY the category name (computer_science, physics, or biology) and a brief explanation (max 50 words).
Format: CATEGORY: explanation"""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse response
            if "computer_science" in result_text.lower():
                category = ScientificCategory.COMPUTER_SCIENCE
            elif "physics" in result_text.lower():
                category = ScientificCategory.PHYSICS
            elif "biology" in result_text.lower():
                category = ScientificCategory.BIOLOGY
            else:
                # Fall back to keyword result
                return keyword_result
            
            # Extract explanation
            parts = result_text.split(":", 1)
            explanation = parts[1].strip() if len(parts) > 1 else result_text
            
            # Higher confidence with LLM validation
            confidence = min(keyword_result.confidence + 0.2, 1.0) if category == keyword_result.category else 0.7
            
            return ClassificationResult(
                category=category,
                confidence=confidence,
                reasoning=f"LLM validation: {explanation}",
                keywords_found=keyword_result.keywords_found
            )
        
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            raise
