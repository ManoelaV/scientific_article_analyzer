"""Critical reviewer for scientific articles"""

import logging
from typing import Optional, List, Dict

from .models import ArticleContent, ClassificationResult, CriticalReview, ScientificCategory

logger = logging.getLogger(__name__)


class CriticalReviewer:
    """Generates critical reviews of scientific articles"""
    
    def __init__(self, openai_api_key: Optional[str] = None, api_base: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize reviewer with optional OpenAI API key"""
        self.openai_api_key = openai_api_key
        self.api_base = api_base
        self.model = model
        self.use_llm = openai_api_key is not None
    
    async def review(
        self, 
        article: ArticleContent, 
        classification: ClassificationResult
    ) -> CriticalReview:
        """
        Generate a critical review of the article.
        
        Args:
            article: Article content to review
            classification: Classification result for context
            
        Returns:
            CriticalReview with summary, aspects, issues, and scores
        """
        if self.use_llm:
            try:
                return await self._review_with_llm(article.text, classification.category)
            except Exception as e:
                logger.warning(f"LLM review failed: {e}. Using rule-based review.")
                return self._review_with_rules(article.text, classification.category)
        
        return self._review_with_rules(article.text, classification.category)
    
    def _review_with_rules(
        self, 
        text: str, 
        category: ScientificCategory
    ) -> CriticalReview:
        """Rule-based review as fallback"""
        
        summary = f"This {category.value} article presents research findings. Detailed analysis requires API access."
        
        positive_aspects = [
            "Article addresses a relevant topic in the field",
            "Structured presentation of information",
            "Contains technical content appropriate for the category"
        ]
        
        potential_issues = [
            "Detailed critique requires full analysis with LLM",
            "Comprehensive evaluation pending"
        ]
        
        # Basic scoring
        base_score = 6.5
        
        return CriticalReview(
            summary=summary,
            positive_aspects=positive_aspects,
            potential_issues=potential_issues,
            overall_score=base_score,
            detailed_scores={
                "methodology": 6.0,
                "clarity": 7.0,
                "originality": 6.5,
                "significance": 6.5
            }
        )
    
    async def _review_with_llm(
        self, 
        text: str, 
        category: ScientificCategory
    ) -> CriticalReview:
        """Use LLM for detailed review"""
        try:
            import openai
            
            # Suporte para APIs customizadas (Ollama, LM Studio, etc.)
            client_kwargs = {"api_key": self.openai_api_key}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            
            client = openai.OpenAI(**client_kwargs)
            
            # Truncate text for API
            text_sample = text[:4000] if len(text) > 4000 else text
            
            # Category-specific criteria
            criteria = self._get_category_criteria(category)
            
            prompt = f"""Provide a critical review of this {category.value} article.

Article:
{text_sample}

Provide:
1. Brief summary (max 100 words)
2. Positive aspects (3-5 points)
3. Potential issues or limitations (3-5 points)
4. Overall quality score (1-10)
5. Scores for: {', '.join(criteria.keys())} (each 1-10)

Format your response as JSON:
{{
  "summary": "...",
  "positive_aspects": ["...", "..."],
  "potential_issues": ["...", "..."],
  "overall_score": 7.5,
  "detailed_scores": {{"methodology": 7, "clarity": 8, ...}}
}}"""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            
            # Remove markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(result_text)
            
            return CriticalReview(
                summary=data.get("summary", ""),
                positive_aspects=data.get("positive_aspects", []),
                potential_issues=data.get("potential_issues", []),
                overall_score=float(data.get("overall_score", 7.0)),
                detailed_scores=data.get("detailed_scores", {}),
                recommendations=data.get("recommendations", [])
            )
        
        except Exception as e:
            logger.error(f"LLM review error: {e}")
            raise
    
    def _get_category_criteria(self, category: ScientificCategory) -> Dict[str, str]:
        """Get evaluation criteria specific to the category"""
        
        if category == ScientificCategory.COMPUTER_SCIENCE:
            return {
                "methodology": "Algorithm design and implementation quality",
                "clarity": "Code and technical explanation clarity",
                "originality": "Novel approach or improvement",
                "performance": "Efficiency and scalability",
                "reproducibility": "Ability to reproduce results"
            }
        
        elif category == ScientificCategory.PHYSICS:
            return {
                "methodology": "Experimental or theoretical rigor",
                "clarity": "Mathematical and conceptual clarity",
                "originality": "Novel physics insights",
                "significance": "Impact on physics understanding",
                "validation": "Quality of validation/verification"
            }
        
        elif category == ScientificCategory.BIOLOGY:
            return {
                "methodology": "Experimental design quality",
                "clarity": "Biological explanation clarity",
                "originality": "Novel biological findings",
                "significance": "Impact on biology/medicine",
                "reproducibility": "Experimental reproducibility"
            }
        
        else:
            return {
                "methodology": "Research methodology quality",
                "clarity": "Overall clarity",
                "originality": "Novelty of approach",
                "significance": "Field significance"
            }
