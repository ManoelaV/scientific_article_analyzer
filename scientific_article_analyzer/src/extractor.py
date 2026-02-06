"""Information extractor for scientific articles"""

import logging
from typing import Optional
import json
import re

from .models import ArticleContent, ExtractedInformation

logger = logging.getLogger(__name__)


class InformationExtractor:
    """Extracts structured information from scientific articles"""
    
    def __init__(self, openai_api_key: Optional[str] = None, api_base: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize extractor with optional OpenAI API key"""
        self.openai_api_key = openai_api_key
        self.api_base = api_base
        self.model = model
        self.use_llm = openai_api_key is not None
    
    async def extract(self, article: ArticleContent) -> ExtractedInformation:
        """
        Extract information from article in standardized format.
        
        Args:
            article: Article content to extract from
            
        Returns:
            ExtractedInformation with problem, solution steps, and conclusion
        """
        if self.use_llm:
            try:
                return await self._extract_with_llm(article.text)
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}. Using rule-based extraction.")
                return self._extract_with_rules(article.text)
        
        return self._extract_with_rules(article.text)
    
    def _extract_with_rules(self, text: str) -> ExtractedInformation:
        """Rule-based extraction as fallback"""
        # Simple heuristics for extraction
        
        # Try to find problem statement
        problem = "Article analysis pending"
        problem_patterns = [
            r"(?:problem|challenge|issue|question)[\s\:]+(.{50,200})",
            r"(?:address|solve|tackle)[\s\:]+(.{50,200})",
            r"(?:aim|goal|objective)[\s\:]+(.{50,200})"
        ]
        
        text_lower = text.lower()
        for pattern in problem_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                problem = match.group(1).strip()
                break
        
        # Extract steps (look for numbered lists or method sections)
        steps = []
        step_patterns = [
            r"(?:step|stage|phase)\s*\d+[\:\s]+(.{30,200})",
            r"\d+\.?\s+(.{30,200})",
            r"(?:first|second|third|finally)[\s\,]+(.{30,200})"
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                steps.extend([m.strip() for m in matches[:5]])  # Max 5 steps
                break
        
        if not steps:
            steps = ["Detailed methodology extraction pending"]
        
        # Try to find conclusion
        conclusion = "Detailed conclusion pending"
        conclusion_patterns = [
            r"(?:conclusion|summary|findings?)[\s\:]+(.{50,300})",
            r"(?:we conclude|in conclusion|to summarize)[\s\:]+(.{50,300})",
            r"(?:results? show|demonstrate)[\s\:]+(.{50,300})"
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                conclusion = match.group(1).strip()
                break
        
        return ExtractedInformation(
            problem=problem[:500],  # Limit length
            solution_steps=steps[:5],  # Max 5 steps
            conclusion=conclusion[:500]
        )
    
    async def _extract_with_llm(self, text: str) -> ExtractedInformation:
        """Use LLM for accurate extraction"""
        try:
            import openai
            
            # Suporte para APIs customizadas (Ollama, LM Studio, etc.)
            client_kwargs = {"api_key": self.openai_api_key}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            
            client = openai.OpenAI(**client_kwargs)
            
            # Truncate text for API
            text_sample = text[:4000] if len(text) > 4000 else text
            
            prompt = f"""Extract the following information from this scientific article:

1. What problem does the article propose to solve? (Be specific and concise, max 150 words)
2. Step by step on how to solve it (List 3-5 main steps/methods)
3. Conclusion (Summary of findings/results, max 150 words)

Article:
{text_sample}

Respond in JSON format:
{{
  "what problem does the article propose to solve?": "...",
  "step by step on how to solve it": ["step 1", "step 2", "step 3"],
  "conclusion": "..."
}}"""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # Remove markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(result_text)
            
            return ExtractedInformation(
                problem=data.get("what problem does the article propose to solve?", ""),
                solution_steps=data.get("step by step on how to solve it", []),
                conclusion=data.get("conclusion", ""),
                raw_data=data
            )
        
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            raise
