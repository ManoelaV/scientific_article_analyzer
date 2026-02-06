"""Article processor for handling different input types (PDF, URL, text)"""

import logging
from typing import Optional
from pathlib import Path
import re

from .models import ArticleContent

logger = logging.getLogger(__name__)


class ArticleProcessor:
    """Processes articles from various sources (PDF, URL, text)"""
    
    def __init__(self):
        self.max_content_length = 50000  # Maximum characters to process
    
    async def process_article(self, input_data: str, input_type: str = "auto") -> ArticleContent:
        """
        Process article from various input types.
        
        Args:
            input_data: Path to PDF, URL, or raw text
            input_type: "pdf", "url", "text", or "auto" for detection
            
        Returns:
            ArticleContent with processed text
        """
        # Auto-detect input type if needed
        if input_type == "auto":
            input_type = self._detect_input_type(input_data)
        
        if input_type == "pdf":
            return await self._process_pdf(input_data)
        elif input_type == "url":
            return await self._process_url(input_data)
        elif input_type == "text":
            return self._process_text(input_data)
        else:
            raise ValueError(f"Unknown input type: {input_type}")
    
    def _detect_input_type(self, input_data: str) -> str:
        """Detect the type of input data"""
        if input_data.startswith(("http://", "https://")):
            return "url"
        elif input_data.endswith(".pdf") or Path(input_data).suffix == ".pdf":
            return "pdf"
        else:
            return "text"
    
    async def _process_pdf(self, pdf_path: str) -> ArticleContent:
        """Extract text from PDF file"""
        try:
            # Try using PyPDF2
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            
            return ArticleContent(
                text=text[:self.max_content_length],
                metadata={"source": pdf_path, "type": "pdf"}
            )
        except ImportError:
            logger.warning("PyPDF2 not installed. Returning placeholder content.")
            return ArticleContent(
                text=f"[PDF content from {pdf_path}]",
                metadata={"source": pdf_path, "type": "pdf", "error": "PyPDF2 not installed"}
            )
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    async def _process_url(self, url: str) -> ArticleContent:
        """Fetch and process article from URL"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Try to extract title
            title = None
            if soup.title:
                title = soup.title.string
            
            return ArticleContent(
                text=text[:self.max_content_length],
                title=title,
                source_url=url,
                metadata={"source": url, "type": "url"}
            )
        except ImportError:
            logger.warning("requests or BeautifulSoup not installed. Returning placeholder content.")
            return ArticleContent(
                text=f"[URL content from {url}]",
                title=f"Article from {url}",
                source_url=url,
                metadata={"source": url, "type": "url", "error": "dependencies not installed"}
            )
        except Exception as e:
            logger.error(f"Error processing URL: {e}")
            raise
    
    def _process_text(self, text: str) -> ArticleContent:
        """Process raw text input"""
        # Try to extract title from first line
        lines = text.strip().split('\n')
        title = None
        if lines and len(lines[0]) < 200:  # Assume first short line is title
            title = lines[0].strip()
        
        return ArticleContent(
            text=text[:self.max_content_length],
            title=title,
            metadata={"type": "text"}
        )
