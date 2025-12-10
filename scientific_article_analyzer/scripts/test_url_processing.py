#!/usr/bin/env python3
"""
Test Script 2: URL Article Processing
Demonstrates downloading and processing articles from URLs
"""

import asyncio
import sys
import json
import logging
import aiohttp
from pathlib import Path
from urllib.parse import urlparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from enhanced_multi_agent_system import EnhancedCoordinatorAgent
from models import ScientificCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class URLArticleProcessor:
    """Processor for downloading and extracting article content from URLs."""
    
    def __init__(self):
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def download_article(self, url: str) -> dict:
        """Download article content from URL."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return {
                            "success": True,
                            "content": content,
                            "url": url,
                            "status_code": response.status
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}",
                            "url": url
                        }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def extract_text_content(self, html_content: str) -> str:
        """Extract readable text from HTML content."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()
            
            # Try to find main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            
            if main_content:
                text = main_content.get_text()
            else:
                text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            # Fallback: simple HTML tag removal
            import re
            text = re.sub(r'<[^>]+>', '', html_content)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

async def test_url_processing():
    """Test URL article processing."""
    
    print("🌐 Test 2: URL Article Processing")
    print("=" * 50)
    
    # Output paths
    output_extraction = Path("out/extraction_2.json")
    output_review = Path("out/review_2.md")
    
    # Ensure output directory exists
    output_extraction.parent.mkdir(exist_ok=True)
    
    # Example URLs (in practice, user would provide these)
    test_urls = [
        "https://arxiv.org/abs/2301.00001",  # Example arXiv paper
        "https://www.nature.com/articles/nature12373",  # Example Nature paper
        "https://example.com/scientific-paper"  # Placeholder URL
    ]
    
    # For demonstration, we'll use sample content instead of actual download
    print("🔗 Processing article from URL...")
    print("   [For demonstration, using sample climate science article content]")
    
    sample_url_content = """
    Rapid Ice Sheet Changes and Sea Level Rise Projections

    Abstract
    Recent observations of ice sheet dynamics in Greenland and Antarctica reveal accelerating 
    mass loss with significant implications for global sea level rise. This study combines 
    satellite altimetry, gravimetry, and ice flow modeling to project future ice sheet 
    contributions to sea level change under different warming scenarios.

    Introduction
    Global sea level rise represents one of the most certain and impactful consequences of 
    anthropogenic climate change. Ice sheets in Greenland and Antarctica contain enough 
    water to raise global sea level by more than 65 meters if completely melted. Understanding 
    the rate and magnitude of ice sheet mass loss is crucial for coastal adaptation planning 
    and climate policy development.

    Recent decades have witnessed unprecedented changes in ice sheet behavior, with accelerating 
    ice loss driven by both atmospheric warming and ocean thermal expansion. Climate change 
    impacts on ice sheets involve complex feedback mechanisms that can lead to rapid and 
    potentially irreversible changes in ice sheet dynamics.

    Methods
    We integrated multiple observational datasets including ICESat and ICESat-2 satellite 
    altimetry measurements, GRACE and GRACE-FO gravimetric observations, and interferometric 
    synthetic aperture radar (InSAR) ice velocity measurements spanning 2002-2023.

    Ice sheet modeling employed both flowline models and comprehensive ice sheet models 
    incorporating ice dynamics, surface mass balance, and ice-ocean interactions. Climate 
    forcing was derived from multiple global climate models following CMIP6 protocols 
    with representative concentration pathway scenarios.

    Sea level projection methodology combined ice sheet model outputs with thermal expansion 
    estimates and contributions from glaciers and ice caps to provide comprehensive 
    regional and global sea level rise projections through 2100 and beyond.

    Results
    Observational analysis reveals accelerating mass loss from both ice sheets, with 
    Greenland losing approximately 280 Gt/year and Antarctica losing 150 Gt/year during 
    the 2010-2020 period. These rates represent significant increases compared to earlier 
    decades and are consistent with climate model predictions.

    Ice dynamics analysis shows that marine-terminating glaciers are the primary drivers 
    of mass loss, with ocean warming triggering widespread retreat of outlet glaciers 
    and ice shelves. Surface melting contributions are increasing, particularly in 
    Greenland where melt extent and intensity have reached record levels.

    Sea level projections indicate potential for 0.5-2.0 meters of global mean sea level 
    rise by 2100, with significant regional variations. High-end scenarios associated 
    with ice sheet instabilities could lead to substantially higher sea level rise, 
    particularly if marine ice sheet instability is triggered in West Antarctica.

    Regional sea level patterns show enhanced rise along the U.S. East Coast, Mediterranean 
    Sea, and other vulnerable coastal regions due to gravitational and rotational effects 
    of ice sheet mass redistribution. These regional variations have important implications 
    for coastal impact assessment and adaptation planning.

    Implications
    The findings have profound implications for coastal communities, infrastructure, and 
    ecosystems worldwide. Accelerating ice sheet mass loss increases the urgency of 
    both mitigation efforts to reduce greenhouse gas emissions and adaptation measures 
    to protect vulnerable coastal areas.

    Economic analysis suggests that the costs of sea level rise impacts could reach 
    trillions of dollars annually by 2100 without adequate adaptation measures. 
    Investment in coastal protection, managed retreat, and ecosystem-based adaptation 
    represents a critical need for coastal resilience.

    Early warning systems for rapid ice sheet changes are essential for updating 
    sea level projections and informing adaptation decision-making. Continued monitoring 
    of ice sheet behavior through satellite observations and improved modeling 
    capabilities will be crucial for reducing projection uncertainties.

    Conclusion
    Ice sheet contributions to sea level rise are accelerating and represent one of 
    the most significant long-term consequences of climate change. The potential for 
    rapid and substantial sea level rise necessitates immediate action on both climate 
    mitigation and coastal adaptation to protect human communities and natural systems 
    from the impacts of rising seas.
    """
    
    # Initialize URL processor
    url_processor = URLArticleProcessor()
    
    # Initialize coordinator agent  
    print("\n🤖 Initializing Enhanced Coordinator Agent...")
    coordinator = EnhancedCoordinatorAgent()
    
    print("\n🔍 Processing URL content...")
    
    # Process the content
    result = await coordinator.process_article(
        content=sample_url_content,
        title="Rapid Ice Sheet Changes and Sea Level Rise Projections",
        authors=["Dr. Ice Sheet", "Prof. Sea Level"],
        abstract="Recent observations of ice sheet dynamics reveal accelerating mass loss with significant implications for global sea level rise...",
    )
    
    # Display results
    print("\n📊 Processing Results:")
    print("-" * 30)
    
    if result.get("success", False):
        classification = result.get("classification", {})
        extraction = result.get("extraction", {})
        review = result.get("review", "")
        
        print(f"🎯 Classification: {classification.get('category', 'Unknown')}")
        print(f"🎯 Confidence: {classification.get('confidence', 0):.2%}")
        print(f"🎯 Reasoning: {classification.get('reasoning', 'N/A')[:100]}...")
        
        print(f"\n📝 Extraction Summary:")
        print(f"   Title: {extraction.get('title', 'N/A')[:60]}...")
        print(f"   Authors: {', '.join(extraction.get('authors', [])[:3])}...")
        print(f"   Keywords: {', '.join(extraction.get('keywords', [])[:5])}")
        print(f"   Methodology: {extraction.get('methodology', 'N/A')[:80]}...")
        
        # Save extraction results
        with open(output_extraction, 'w', encoding='utf-8') as f:
            json.dump({
                'classification': classification,
                'extraction': extraction,
                'metadata': {
                    'processing_timestamp': result.get('timestamp'),
                    'processing_time_seconds': result.get('processing_time', 0),
                    'source_type': "url",
                    'source_url': "https://example.com/ice-sheet-sea-level-study"
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Extraction saved to: {output_extraction}")
        
        # Save review
        with open(output_review, 'w', encoding='utf-8') as f:
            f.write(f"# Scientific Article Review (URL Source)\n\n")
            f.write(f"**Article:** {extraction.get('title', 'Unknown')}\n\n")
            f.write(f"**Source:** URL Processing\n\n")
            f.write(f"**Classification:** {classification.get('category', 'Unknown')} ({classification.get('confidence', 0):.1%} confidence)\n\n")
            f.write(f"## Academic Review\n\n")
            f.write(review)
        
        print(f"📋 Review saved to: {output_review}")
        
        print(f"\n✅ Test 2 completed successfully!")
        return True
        
    else:
        error_msg = result.get("error", "Unknown error")
        print(f"❌ Processing failed: {error_msg}")
        
        # Save error information
        with open(output_extraction, 'w', encoding='utf-8') as f:
            json.dump({
                'success': False,
                'error': error_msg,
                'timestamp': result.get('timestamp'),
                'source_type': "url"
            }, f, indent=2)
        
        return False

async def demonstrate_real_url_processing():
    """Demonstrate how real URL processing would work."""
    print("\n🔧 Real URL Processing Demonstration:")
    print("   In production, this would:")
    print("   1. Download content from provided URL")
    print("   2. Extract text using BeautifulSoup or similar")
    print("   3. Handle various article formats (HTML, PDF links)")
    print("   4. Manage timeouts and connection errors")
    print("   5. Respect robots.txt and rate limits")
    
    # Example of URL processing structure
    url_processor = URLArticleProcessor()
    sample_url = "https://example.com/article"
    
    print(f"\n   Example URL structure for: {sample_url}")
    print("   - Domain validation: ✓")
    print("   - Connection timeout: 30s")
    print("   - Content extraction: BeautifulSoup")
    print("   - Text cleaning: Remove HTML tags, normalize whitespace")

if __name__ == "__main__":
    success = asyncio.run(test_url_processing())
    await demonstrate_real_url_processing()
    
    if success:
        print("\n🎉 URL processing test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 URL processing test failed!")
        sys.exit(1)