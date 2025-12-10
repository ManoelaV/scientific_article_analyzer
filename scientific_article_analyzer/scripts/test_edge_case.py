#!/usr/bin/env python3
"""
Test Script 3: Edge Case Processing
Demonstrates handling of articles outside the three main scientific areas
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

logging.basicConfig(level=logging.INFO)
logger = logger.getLogger(__name__)

async def test_edge_case_processing():
    """Test processing of articles outside the main scientific areas."""
    
    print("⚠️  Test 3: Edge Case Processing")
    print("=" * 50)
    
    # Output paths
    output_extraction = Path("out/extraction_3.json")
    output_review = Path("out/review_3.md")
    
    # Ensure output directory exists
    output_extraction.parent.mkdir(exist_ok=True)
    
    print("🔍 Processing article outside main scientific areas...")
    print("   Testing system response to non-matching content")
    
    # Sample content from a different scientific area (Psychology/Sociology)
    edge_case_content = """
    Social Media Impact on Adolescent Mental Health: A Longitudinal Study

    Abstract
    This longitudinal study examines the relationship between social media usage patterns 
    and mental health outcomes in adolescents over a three-year period. Data from 2,500 
    participants aged 13-17 reveals significant correlations between excessive social 
    media use and increased rates of depression, anxiety, and sleep disorders.

    Introduction
    The proliferation of social media platforms has fundamentally altered how adolescents 
    interact, communicate, and perceive themselves and others. While these platforms offer 
    opportunities for connection and self-expression, growing concerns exist about their 
    impact on mental health and psychological well-being.

    Recent studies have suggested links between social media use and various mental health 
    issues, but many have been cross-sectional or focused on limited populations. This 
    study addresses these limitations by following a diverse cohort of adolescents over 
    multiple years to establish temporal relationships and identify causal mechanisms.

    The developmental period of adolescence coincides with increased vulnerability to 
    mental health challenges and heightened sensitivity to social influences. Understanding 
    how digital social environments affect this critical developmental stage has important 
    implications for public health policy and clinical practice.

    Methods
    We conducted a longitudinal cohort study following 2,500 adolescents (ages 13-17) 
    from diverse socioeconomic and ethnic backgrounds across urban, suburban, and rural 
    communities. Participants were recruited through schools, community organizations, 
    and healthcare providers to ensure representative sampling.

    Data collection occurred at six-month intervals over three years using validated 
    psychological assessment instruments including the PHQ-9 for depression screening, 
    GAD-7 for anxiety assessment, and the Pittsburgh Sleep Quality Index. Social media 
    usage was tracked through self-report surveys and, with participant consent, 
    objective smartphone usage data.

    Statistical analysis employed multilevel modeling to account for repeated measurements 
    within individuals and potential clustering effects within schools and communities. 
    Mediation analysis investigated potential mechanisms linking social media use to 
    mental health outcomes, including sleep disruption, cyberbullying exposure, and 
    social comparison processes.

    Ethical considerations included robust privacy protections, ongoing consent processes, 
    and referral pathways for participants showing signs of mental health distress. 
    The study protocol was approved by institutional review boards and included youth 
    advisory groups in study design and implementation.

    Results
    Participants demonstrated high levels of social media engagement, with average 
    daily usage exceeding 4 hours among 60% of the sample. Usage patterns varied 
    significantly by platform, demographic characteristics, and time of day, with 
    peak usage occurring in evening hours often coinciding with recommended sleep times.

    Mental health outcomes showed concerning trends over the study period, with 23% 
    of participants experiencing clinically significant depression symptoms and 31% 
    reporting moderate to severe anxiety by the final assessment. Sleep quality 
    declined significantly, with average sleep duration decreasing by 45 minutes 
    per night over the study period.

    Correlation analysis revealed significant associations between heavy social media 
    use (>6 hours daily) and increased depression scores (r=0.34, p<0.001), higher 
    anxiety levels (r=0.28, p<0.001), and poorer sleep quality (r=-0.41, p<0.001). 
    These associations remained significant after controlling for demographic factors, 
    baseline mental health status, and other potential confounders.

    Mediation analysis identified several key pathways linking social media use to 
    mental health outcomes. Sleep disruption accounted for approximately 35% of the 
    association between usage and depression symptoms. Cyberbullying exposure mediated 
    20% of the relationship with anxiety, while social comparison processes contributed 
    to both depression and anxiety outcomes.

    Gender differences emerged in both usage patterns and mental health impacts, with 
    female participants showing higher rates of appearance-focused social comparison 
    and stronger associations between usage and mood symptoms. Male participants 
    demonstrated different usage patterns but similar overall mental health risks.

    Platform-specific analysis revealed varying risk profiles, with image-focused 
    platforms showing stronger associations with body image concerns and mood symptoms, 
    while text-based platforms were more associated with social comparison and 
    academic stress.

    Discussion
    The findings provide robust evidence for significant mental health risks associated 
    with excessive social media use during adolescence. The longitudinal design enables 
    stronger causal inferences than previous cross-sectional studies and identifies 
    specific mechanisms through which social media impacts psychological well-being.

    The results have important implications for parents, educators, healthcare providers, 
    and policymakers. Evidence-based guidelines for healthy social media use during 
    adolescence are needed, along with interventions targeting identified risk mechanisms 
    such as sleep hygiene and social comparison processes.

    Clinical implications include the need for routine screening of social media habits 
    in adolescent mental health assessments and the development of targeted interventions 
    addressing digital wellness. Healthcare providers should be prepared to discuss 
    social media use as a potential contributing factor to mental health concerns.

    Policy recommendations include consideration of age verification requirements, 
    platform design modifications to reduce harmful features, and digital literacy 
    education in school curricula. Collaboration between technology companies, 
    researchers, and public health officials is essential for developing effective 
    protective measures.

    Limitations include reliance on self-reported usage data for some participants, 
    potential selection bias in the volunteer sample, and the rapidly evolving nature 
    of social media platforms that may limit generalizability of findings over time.

    Future research should investigate protective factors that may mitigate social 
    media risks, examine longer-term outcomes into young adulthood, and evaluate 
    the effectiveness of intervention strategies designed to promote healthy digital 
    media habits among adolescents.

    Conclusion
    This longitudinal study provides compelling evidence that excessive social media 
    use during adolescence is associated with significant mental health risks including 
    depression, anxiety, and sleep disorders. The identification of specific risk 
    mechanisms offers targets for intervention and prevention efforts. Addressing 
    the mental health implications of social media use requires coordinated efforts 
    across multiple stakeholders to protect adolescent well-being in the digital age.
    """
    
    # Initialize coordinator agent
    print("\n🤖 Initializing Enhanced Coordinator Agent...")
    coordinator = EnhancedCoordinatorAgent()
    
    print("\n🔍 Processing edge case content...")
    print("   Content area: Psychology/Mental Health (outside ML/Climate/Biotech)")
    
    # Process the content
    result = await coordinator.process_article(
        content=edge_case_content,
        title="Social Media Impact on Adolescent Mental Health: A Longitudinal Study",
        authors=["Dr. Psychology Research", "Prof. Mental Health"],
        abstract="This longitudinal study examines the relationship between social media usage patterns and mental health outcomes in adolescents...",
    )
    
    # Display results
    print("\n📊 Edge Case Processing Results:")
    print("-" * 40)
    
    if result.get("success", False):
        classification = result.get("classification", {})
        extraction = result.get("extraction", {})
        review = result.get("review", "")
        
        predicted_category = classification.get('category', 'Unknown')
        confidence = classification.get('confidence', 0)
        reasoning = classification.get('reasoning', 'N/A')
        
        print(f"🎯 Predicted Category: {predicted_category}")
        print(f"🎯 Confidence: {confidence:.2%}")
        print(f"🎯 System Reasoning: {reasoning[:150]}...")
        
        # Analyze edge case handling
        if predicted_category in [category.value for category in ScientificCategory]:
            print(f"\n⚠️  Edge Case Analysis:")
            print(f"   System incorrectly classified psychology article as {predicted_category}")
            print(f"   This demonstrates the need for 'OTHER' or 'UNKNOWN' category handling")
            print(f"   Confidence level: {confidence:.1%} - {'LOW' if confidence < 0.5 else 'MEDIUM' if confidence < 0.8 else 'HIGH'}")
        else:
            print(f"\n✅ Edge Case Handling:")
            print(f"   System correctly identified content as outside main categories")
            print(f"   Appropriate handling of non-matching content")
        
        print(f"\n📝 Extraction Summary:")
        print(f"   Title: {extraction.get('title', 'N/A')[:60]}...")
        print(f"   Authors: {', '.join(extraction.get('authors', [])[:3])}...")
        print(f"   Keywords: {', '.join(extraction.get('keywords', [])[:5])}")
        print(f"   Research Domain: Psychology/Mental Health")
        
        # Save extraction results with edge case analysis
        edge_case_analysis = {
            "is_edge_case": True,
            "expected_behavior": "System should either classify with low confidence or identify as 'OTHER'",
            "actual_behavior": f"Classified as {predicted_category} with {confidence:.2%} confidence",
            "recommendation": "Implement threshold-based rejection for low-confidence classifications",
            "content_domain": "Psychology/Mental Health"
        }
        
        with open(output_extraction, 'w', encoding='utf-8') as f:
            json.dump({
                'classification': classification,
                'extraction': extraction,
                'edge_case_analysis': edge_case_analysis,
                'metadata': {
                    'processing_timestamp': result.get('timestamp'),
                    'processing_time_seconds': result.get('processing_time', 0),
                    'test_type': "edge_case",
                    'content_domain': "psychology_mental_health"
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Extraction saved to: {output_extraction}")
        
        # Save review with edge case notes
        with open(output_review, 'w', encoding='utf-8') as f:
            f.write(f"# Scientific Article Review (Edge Case)\n\n")
            f.write(f"**Article:** {extraction.get('title', 'Unknown')}\n\n")
            f.write(f"**Test Type:** Edge Case Processing\n\n")
            f.write(f"**Content Domain:** Psychology/Mental Health (Outside Target Areas)\n\n")
            f.write(f"**Classification Result:** {predicted_category} ({confidence:.1%} confidence)\n\n")
            f.write(f"## Edge Case Analysis\n\n")
            f.write(f"This article tests the system's ability to handle content outside the three main scientific areas.\n\n")
            f.write(f"**Expected Behavior:** The system should either:\n")
            f.write(f"- Classify with low confidence indicating uncertainty\n")
            f.write(f"- Reject classification if confidence is below threshold\n")
            f.write(f"- Identify content as belonging to 'OTHER' category\n\n")
            f.write(f"**Actual Behavior:** {edge_case_analysis['actual_behavior']}\n\n")
            f.write(f"**Recommendation:** {edge_case_analysis['recommendation']}\n\n")
            f.write(f"## Academic Review\n\n")
            f.write(review)
            f.write(f"\n\n## System Performance Notes\n\n")
            f.write(f"The system's handling of this edge case provides insights into classification robustness and the need for appropriate confidence thresholds or explicit 'other' category handling in production deployments.\n")
        
        print(f"📋 Review saved to: {output_review}")
        
        print(f"\n✅ Test 3 completed successfully!")
        
        # Additional edge case recommendations
        print(f"\n🔧 Edge Case Handling Recommendations:")
        print(f"   1. Implement confidence threshold (e.g., <50% = 'UNCERTAIN')")
        print(f"   2. Add 'OTHER' category for non-matching content")
        print(f"   3. Provide reasoning for low-confidence classifications")
        print(f"   4. Log edge cases for model improvement")
        
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
                'test_type': "edge_case",
                'edge_case_analysis': {
                    "is_edge_case": True,
                    "error_handling": "System failed to process edge case content",
                    "recommendation": "Improve error handling for unexpected content types"
                }
            }, f, indent=2)
        
        return False

def demonstrate_edge_case_strategies():
    """Demonstrate strategies for handling edge cases in production."""
    print("\n🔧 Edge Case Handling Strategies:")
    print("   1. Confidence Thresholding:")
    print("      - Classifications below 50% confidence → 'UNCERTAIN'")
    print("      - Provide explanation for uncertainty")
    
    print("\n   2. Multiple Classification Approaches:")
    print("      - Primary: Target categories (ML, Climate, Biotech)")
    print("      - Secondary: Broader scientific areas")
    print("      - Fallback: 'OTHER' with domain identification")
    
    print("\n   3. Uncertainty Quantification:")
    print("      - Confidence intervals for predictions")
    print("      - Ensemble model disagreement detection")
    print("      - Outlier detection in embedding space")
    
    print("\n   4. Human-in-the-Loop:")
    print("      - Flag uncertain cases for human review")
    print("      - Collect feedback for model improvement")
    print("      - Gradual expansion of supported categories")

if __name__ == "__main__":
    success = asyncio.run(test_edge_case_processing())
    demonstrate_edge_case_strategies()
    
    if success:
        print("\n🎉 Edge case processing test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Edge case processing test failed!")
        sys.exit(1)