#!/usr/bin/env python3
"""
Test Script 1: PDF Processing and Classification
Demonstrates article classification and metadata extraction from PDF
"""

import asyncio
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from enhanced_multi_agent_system import EnhancedCoordinatorAgent
from models import ScientificCategory
from utils.pdf_processor import PDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pdf_processing():
    """Test PDF processing, classification, and extraction."""
    
    print("📄 Test 1: PDF Classification and Extraction")
    print("=" * 50)
    
    # Input and output paths
    input_pdf = Path("samples/input_article_1.pdf")
    output_extraction = Path("out/extraction_1.json") 
    output_review = Path("out/review_1.md")
    
    # Ensure output directory exists
    output_extraction.parent.mkdir(exist_ok=True)
    
    # Check if input file exists
    if not input_pdf.exists():
        print(f"❌ Input file not found: {input_pdf}")
        print("   Creating sample PDF content for demonstration...")
        
        # Create sample PDF content (placeholder - would need actual PDF creation)
        print("   [In real scenario, a sample scientific article PDF would be provided]")
        
        # For demonstration, we'll process sample text content
        sample_content = """
        Deep Learning for Medical Image Analysis: A Comprehensive Review

        Abstract
        This paper presents a comprehensive review of deep learning applications in medical image analysis. 
        We examine recent advances in convolutional neural networks, transfer learning approaches, and 
        specialized architectures for medical imaging tasks including diagnosis, segmentation, and 
        treatment planning. Our analysis covers applications in radiology, pathology, and clinical 
        decision support systems.

        Introduction
        Medical image analysis has been revolutionized by the advent of deep learning technologies. 
        Traditional computer vision approaches often struggled with the complexity and variability 
        of medical images, but deep learning methods have demonstrated remarkable success across 
        various clinical applications.

        The integration of artificial intelligence in healthcare represents one of the most promising 
        developments in modern medicine. Machine learning algorithms, particularly deep neural networks, 
        have shown exceptional performance in tasks such as image classification, object detection, 
        and semantic segmentation when applied to medical imaging data.

        Methodology
        We conducted a systematic review of peer-reviewed literature published between 2018 and 2023, 
        focusing on deep learning applications in medical image analysis. Our search strategy included 
        major databases such as PubMed, IEEE Xplore, and ACM Digital Library.

        The analysis framework encompasses multiple aspects of deep learning implementation including 
        network architectures, training strategies, validation methodologies, and clinical validation 
        studies. We categorized applications by medical specialty and imaging modality to provide 
        comprehensive coverage of the field.

        Results
        Our review identified significant advances in several key areas. Convolutional neural networks 
        have achieved human-level performance in many diagnostic tasks, particularly in radiology and 
        dermatology. Transfer learning approaches have enabled effective model development even with 
        limited medical imaging datasets.

        Novel architectures such as attention-based models and transformer networks are showing promise 
        for complex medical imaging tasks. These approaches can handle multi-modal data integration 
        and provide interpretable results crucial for clinical adoption.

        Conclusion
        Deep learning has transformed medical image analysis, offering unprecedented accuracy and 
        efficiency in clinical applications. Future research should focus on improving model 
        interpretability, addressing data bias, and ensuring robust performance across diverse 
        patient populations and clinical settings.
        """
        
        # Initialize coordinator agent
        print("\n🤖 Initializing Enhanced Coordinator Agent...")
        coordinator = EnhancedCoordinatorAgent()
        
        print("\n🔍 Processing article content...")
        
        # Process the content
        result = await coordinator.process_article(
            content=sample_content,
            title="Deep Learning for Medical Image Analysis: A Comprehensive Review",
            authors=["Dr. Medical AI", "Prof. Image Analysis"],
            abstract="This paper presents a comprehensive review of deep learning applications in medical image analysis...",
        )
        
    else:
        # Process actual PDF file
        print(f"📖 Processing PDF: {input_pdf}")
        
        # Extract text from PDF
        pdf_processor = PDFProcessor()
        pdf_content = pdf_processor.extract_text(str(input_pdf))
        
        if not pdf_content:
            print("❌ Failed to extract text from PDF")
            return False
            
        print(f"✅ Extracted {len(pdf_content)} characters from PDF")
        
        # Initialize coordinator agent
        print("\n🤖 Initializing Enhanced Coordinator Agent...")
        coordinator = EnhancedCoordinatorAgent()
        
        print("\n🔍 Processing PDF content...")
        
        # Process the PDF content
        result = await coordinator.process_article(content=pdf_content)
    
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
                    'source_file': str(input_pdf) if input_pdf.exists() else "sample_content"
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Extraction saved to: {output_extraction}")
        
        # Save review
        with open(output_review, 'w', encoding='utf-8') as f:
            f.write(f"# Scientific Article Review\n\n")
            f.write(f"**Article:** {extraction.get('title', 'Unknown')}\n\n")
            f.write(f"**Classification:** {classification.get('category', 'Unknown')} ({classification.get('confidence', 0):.1%} confidence)\n\n")
            f.write(f"## Academic Review\n\n")
            f.write(review)
        
        print(f"📋 Review saved to: {output_review}")
        
        print(f"\n✅ Test 1 completed successfully!")
        return True
        
    else:
        error_msg = result.get("error", "Unknown error")
        print(f"❌ Processing failed: {error_msg}")
        
        # Save error information
        with open(output_extraction, 'w', encoding='utf-8') as f:
            json.dump({
                'success': False,
                'error': error_msg,
                'timestamp': result.get('timestamp')
            }, f, indent=2)
        
        return False

if __name__ == "__main__":
    success = asyncio.run(test_pdf_processing())
    if success:
        print("\n🎉 PDF processing test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 PDF processing test failed!")
        sys.exit(1)