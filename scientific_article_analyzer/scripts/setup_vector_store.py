#!/usr/bin/env python3
"""
Setup Vector Store Script
Initializes and populates the vector store with reference articles
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vector_store.advanced_store import AdvancedVectorStore, VectorStoreConfig, EmbeddingProvider, ChunkingConfig, ChunkingStrategy
from src.models import ArticleContent, ScientificCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_vector_store():
    """Setup and populate vector store with reference articles."""
    
    print("🔧 Initializing Advanced Vector Store...")
    
    # Create configuration
    config = VectorStoreConfig(
        embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
        embedding_model="all-MiniLM-L6-v2",
        chunking_config=ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=1000,
            overlap=200,
            min_chunk_size=100,
            max_chunk_size=2000
        ),
        storage_path="./data/advanced_vector_store",
        use_faiss=True,
        similarity_threshold=0.7
    )
    
    # Initialize vector store
    vector_store = AdvancedVectorStore(config)
    
    print("📚 Adding reference articles...")
    
    # Reference articles for each category (3 per category = 9 total)
    reference_articles = {
        ScientificCategory.MACHINE_LEARNING: [
            {
                "title": "Deep Neural Networks for Advanced Pattern Recognition",
                "abstract": "This paper presents novel deep learning architectures for complex pattern recognition tasks, demonstrating significant improvements over traditional methods through attention mechanisms and architectural innovations.",
                "authors": ["Dr. Maria Silva", "Prof. João Santos", "Dr. Ana Costa"],
                "keywords": ["deep learning", "neural networks", "pattern recognition", "attention mechanisms", "computer vision"],
                "content": """
                Deep neural networks have revolutionized the field of pattern recognition through their ability to learn hierarchical representations from raw data. This research introduces novel architectural improvements that enhance both accuracy and computational efficiency. Our methodology combines convolutional layers with attention mechanisms, creating a hybrid architecture that outperforms existing state-of-the-art models.

                The proposed architecture incorporates multi-scale feature extraction through dilated convolutions, enabling the network to capture both fine-grained details and global context simultaneously. We introduce a novel attention module that dynamically weights feature channels based on their relevance to the classification task, significantly improving model interpretability and performance.

                Experimental validation was conducted on multiple benchmark datasets including ImageNet, CIFAR-100, and custom industrial datasets. The methodology involves systematic architecture search combined with principled design choices based on theoretical insights from information theory and cognitive science. Training procedures include advanced data augmentation techniques, progressive learning rates, and regularization strategies.

                Results demonstrate significant improvements across all tested scenarios, with accuracy gains of up to 15% compared to previous methods. The proposed architecture shows particular strength in handling complex, high-dimensional data while maintaining computational efficiency. Ablation studies confirm the importance of each architectural component, with the attention mechanism contributing most significantly to performance improvements.

                Key contributions include: (1) Novel attention-based convolution blocks that adaptively focus on relevant features, (2) Improved training stability through adaptive normalization techniques, (3) Comprehensive evaluation demonstrating superior performance across diverse domains, (4) Theoretical analysis explaining the effectiveness of the proposed approach.

                The approach demonstrates excellent generalization capabilities across different data domains, from natural images to medical imaging and satellite imagery. Performance analysis reveals consistent improvements in both accuracy and computational efficiency, making the method suitable for real-world deployment scenarios.

                Future work will explore the application of these architectural innovations to other domains such as natural language processing and multimodal learning. The modular design of our approach facilitates easy adaptation to different problem domains and data types.

                In conclusion, this work advances the state-of-the-art in deep learning architectures and provides a foundation for future research in neural network design, offering both theoretical insights and practical improvements for pattern recognition tasks.
                """
            },
            {
                "title": "Reinforcement Learning for Autonomous System Control",
                "abstract": "A comprehensive study on applying deep reinforcement learning algorithms to autonomous system control, with applications in robotics, autonomous vehicles, and industrial automation.",
                "authors": ["Prof. Carlos Roboitca", "Dr. Elena Autonoma"],
                "keywords": ["reinforcement learning", "autonomous systems", "robotics", "control theory", "artificial intelligence"],
                "content": """
                Reinforcement learning has emerged as a powerful paradigm for autonomous system control, enabling agents to learn optimal policies through interaction with complex environments. This research presents a comprehensive framework for applying deep reinforcement learning algorithms to real-world autonomous systems.

                Our methodology integrates model-free reinforcement learning with advanced control theory, creating hybrid approaches that combine the adaptability of RL with the stability guarantees of classical control. We develop novel algorithms that can handle continuous action spaces, partial observability, and multi-objective optimization scenarios common in autonomous systems.

                The experimental framework encompasses three major application domains: robotic manipulation, autonomous vehicle navigation, and industrial process control. Each domain presents unique challenges requiring specialized adaptations of the core RL algorithms. We implement and evaluate several state-of-the-art algorithms including PPO, SAC, and novel variants designed specifically for control applications.

                Results demonstrate significant improvements in control performance across all tested scenarios. In robotic manipulation tasks, our approach achieves 40% faster task completion with 60% fewer failures compared to traditional control methods. Autonomous navigation experiments show improved safety margins and more efficient path planning in complex environments.

                Key innovations include: (1) Hybrid RL-control architectures that maintain safety guarantees, (2) Novel reward shaping techniques for complex multi-objective scenarios, (3) Transfer learning methods enabling rapid adaptation to new environments, (4) Real-time implementation strategies for resource-constrained systems.

                Safety considerations are paramount in autonomous systems, and our approach incorporates formal verification methods to ensure system reliability. We develop safety-aware training procedures that prevent the agent from exploring potentially dangerous states during learning, while still enabling effective exploration of the policy space.

                The framework's modularity allows for easy adaptation to different autonomous systems and control objectives. Extensive simulation and real-world testing validate the approach's effectiveness across diverse operational conditions and system configurations.

                Future research directions include extending the framework to multi-agent scenarios, incorporating human-in-the-loop learning, and developing more sophisticated safety verification methods for autonomous systems operating in unpredictable environments.
                """
            },
            {
                "title": "Transformer Architectures for Natural Language Understanding",
                "abstract": "An in-depth analysis of transformer-based models for natural language understanding tasks, including novel attention mechanisms and pre-training strategies for improved performance.",
                "authors": ["Dr. Linguista Nova", "Prof. Processamento Natural"],
                "keywords": ["transformers", "natural language processing", "attention mechanisms", "pre-training", "language models"],
                "content": """
                Transformer architectures have fundamentally transformed natural language processing, establishing new state-of-the-art results across virtually all NLP tasks. This research investigates novel modifications to transformer architectures specifically designed for enhanced natural language understanding capabilities.

                Our approach focuses on improving the attention mechanism's ability to capture long-range dependencies and semantic relationships within text. We introduce sparse attention patterns that reduce computational complexity while maintaining or improving performance on understanding tasks. Additionally, we develop new positional encoding schemes that better handle variable-length sequences and improve generalization.

                The methodology encompasses comprehensive pre-training strategies on large-scale corpora, followed by careful fine-tuning on downstream tasks. We experiment with different pre-training objectives, including masked language modeling variations, next sentence prediction alternatives, and novel self-supervised tasks designed to enhance semantic understanding.

                Experimental evaluation covers a wide range of NLP benchmarks including GLUE, SuperGLUE, and domain-specific understanding tasks. We conduct extensive ablation studies to understand the contribution of each architectural component and training strategy. The evaluation methodology includes both automatic metrics and human evaluation to assess the quality of language understanding.

                Results show significant improvements across multiple understanding tasks, with particular strengths in tasks requiring complex reasoning and long-range dependency modeling. Our approach achieves new state-of-the-art results on several challenging benchmarks while requiring fewer parameters than comparable models.

                Key contributions include: (1) Novel sparse attention mechanisms that improve efficiency without sacrificing performance, (2) Enhanced positional encoding schemes for better sequence modeling, (3) Improved pre-training strategies that lead to better downstream performance, (4) Comprehensive analysis of what linguistic phenomena different model components capture.

                The research provides insights into the mechanisms underlying transformer success in NLP, offering both theoretical understanding and practical improvements. We analyze the learned representations to understand how different architectural choices affect the model's ability to capture syntactic and semantic information.

                Scalability analysis demonstrates that our improvements maintain their effectiveness across different model sizes, from compact models suitable for mobile deployment to large-scale models for research applications. The approach shows promising results for multilingual understanding tasks and cross-lingual transfer learning.

                Future work will explore the application of these architectural innovations to multimodal understanding tasks and investigate more efficient training procedures for very large language models.
                """
            }
        ],
        
        ScientificCategory.CLIMATE_SCIENCE: [
            {
                "title": "Climate Change Impacts on Arctic Ice Coverage: A Comprehensive Satellite Analysis",
                "abstract": "This study examines three decades of Arctic ice coverage changes using advanced satellite remote sensing data, revealing accelerating ice loss patterns and their implications for global climate systems.",
                "authors": ["Dr. Anna Klimova", "Prof. Robert Arctic", "Dr. Sarah Glacial"],
                "keywords": ["climate change", "arctic ice", "satellite remote sensing", "global warming", "ice coverage trends"],
                "content": """
                The Arctic region has experienced unprecedented changes in ice coverage over the past three decades, with satellite observations revealing alarming trends in ice loss that have profound implications for global climate systems. This comprehensive study analyzes multi-temporal satellite data to quantify and understand the complex patterns of Arctic ice decline.

                Our methodology employs state-of-the-art remote sensing techniques, combining data from multiple satellite missions including Landsat, MODIS, and synthetic aperture radar systems. We developed advanced image processing algorithms capable of handling the unique challenges of Arctic remote sensing, including polar night conditions, cloud cover, and atmospheric interference.

                The analytical framework incorporates machine learning approaches for automated ice classification, time series analysis for trend detection, and climate modeling for impact assessment. We processed over 50,000 satellite images spanning 1990-2023, creating the most comprehensive Arctic ice database to date.

                Results reveal a consistent and accelerating trend of ice loss, with annual ice coverage declining by approximately 3.2% per decade. The most significant changes occur during summer months, when ice coverage reaches historical lows with increasing frequency. Spatial analysis shows that ice loss is not uniform, with the greatest changes occurring in specific regions of the Arctic Ocean.

                Temperature correlations demonstrate strong relationships between warming trends and ice loss rates, with feedback mechanisms amplifying the effects. We identified critical temperature thresholds beyond which ice loss accelerates dramatically, providing important insights for climate prediction models.

                Key findings include: (1) Accelerating ice loss rates in recent years, particularly post-2010, (2) Strong correlation with global temperature anomalies and Arctic amplification effects, (3) Regional variations in ice loss patterns linked to ocean currents and atmospheric circulation, (4) Significant implications for global sea level rise projections and regional weather patterns.

                The research reveals complex feedback mechanisms that amplify warming effects in polar regions, including the albedo effect and changes in ocean circulation patterns. These findings have critical implications for global climate models and sea level rise projections.

                Ecosystem impacts are substantial, affecting polar bear populations, marine food webs, and indigenous communities that depend on sea ice for traditional ways of life. The study documents unprecedented changes in ice thickness and seasonal duration that threaten Arctic biodiversity.

                The implications extend far beyond the Arctic region, affecting global climate patterns through disruptions to the thermohaline circulation, changes in storm tracks, and modifications to the jet stream. Our analysis provides crucial data for climate policy formulation and adaptation strategies.

                Future research will focus on improving predictive models for Arctic ice changes and investigating the potential for irreversible tipping points in the Arctic climate system. The findings underscore the urgent need for global climate action to mitigate further ice loss.
                """
            },
            {
                "title": "Ocean Acidification Trends and Marine Ecosystem Responses",
                "abstract": "Long-term monitoring of ocean pH changes reveals significant acidification trends with measurable impacts on marine calcifying organisms and ecosystem dynamics across multiple ocean basins.",
                "authors": ["Dr. Marina Oceanica", "Prof. Acidez Maritima"],
                "keywords": ["ocean acidification", "marine ecosystems", "pH monitoring", "calcifying organisms", "carbon cycle"],
                "content": """
                Ocean acidification, often called the 'other CO2 problem,' represents one of the most significant threats to marine ecosystems in the modern era. This research presents comprehensive analysis of long-term ocean pH monitoring data and its correlation with marine ecosystem health indicators across major ocean basins.

                Our methodology combines extensive field measurements, laboratory experiments, and ecosystem modeling to understand the complex interactions between changing ocean chemistry and marine life. We established monitoring stations across the Atlantic, Pacific, and Indian Oceans, collecting data on pH, carbonate saturation, and biological indicators over a 15-year period.

                Laboratory experiments focused on the responses of key calcifying organisms including corals, mollusks, and calcifying plankton to varying pH conditions. We conducted controlled exposure studies to determine threshold levels and adaptation capacities of different species to acidification stress.

                The data reveals significant acidification trends across all monitored regions, with pH decreasing by an average of 0.1 units over the study period. This change, while seemingly small, represents a 26% increase in acidity with profound biological implications. Regional variations show the most severe acidification in areas with high CO2 emissions and limited buffering capacity.

                Biological impacts are already measurable across multiple taxonomic groups. Calcifying organisms show reduced shell and skeleton formation, with some species exhibiting up to 30% reduction in calcification rates. Coral reef systems demonstrate increased bleaching susceptibility and reduced recovery rates in areas with enhanced acidification.

                Key findings include: (1) Accelerating acidification rates in coastal regions due to combined atmospheric CO2 and local pollution sources, (2) Species-specific responses ranging from high sensitivity to surprising resilience, (3) Ecosystem-level changes including altered food web dynamics and biodiversity shifts, (4) Economic implications for fisheries and marine aquaculture industries.

                The research identifies critical pH thresholds for different ecosystem components, providing valuable information for ecosystem management and conservation strategies. We document cascading effects through marine food webs, from microscopic plankton to large predatory fish.

                Regional hotspots of acidification include upwelling zones, coastal areas with heavy industrial activity, and regions with limited water circulation. These areas serve as natural laboratories for understanding future ocean conditions and ecosystem responses.

                Mitigation strategies are evaluated, including both global approaches focused on CO2 reduction and local interventions such as alkalinity enhancement and nutrient management. The effectiveness of marine protected areas in building ecosystem resilience to acidification is assessed through comparative studies.

                The findings have immediate implications for marine resource management, coastal communities dependent on marine ecosystems, and global carbon cycle understanding. Our work contributes essential data for international climate negotiations and marine conservation policy development.

                Future research priorities include developing early warning systems for acidification events, investigating adaptation mechanisms in marine organisms, and improving predictive models for ecosystem responses to continued ocean chemistry changes.
                """
            },
            {
                "title": "Extreme Weather Events and Climate Attribution Studies",
                "abstract": "Advanced statistical analysis of extreme weather patterns reveals clear fingerprints of anthropogenic climate change, with implications for risk assessment and adaptation planning in vulnerable regions.",
                "authors": ["Prof. Extremo Clima", "Dr. Atribuição Temporal"],
                "keywords": ["extreme weather", "climate attribution", "statistical analysis", "risk assessment", "climate change"],
                "content": """
                The attribution of extreme weather events to anthropogenic climate change represents one of the most challenging and societally relevant aspects of modern climate science. This research develops advanced statistical methodologies for detecting and attributing changes in extreme weather patterns to human activities versus natural variability.

                Our methodological framework combines observational data analysis, climate model ensembles, and cutting-edge statistical techniques including Bayesian attribution methods and machine learning approaches. We analyze multiple types of extreme events including heatwaves, droughts, floods, and severe storms across different geographical regions and time scales.

                The statistical approach employs both frequentist and Bayesian frameworks for event attribution, allowing for quantitative assessment of the probability that specific extreme events were influenced by anthropogenic climate change. We develop new metrics for characterizing the magnitude and likelihood of attribution while accounting for natural climate variability.

                Observational datasets span over 150 years of meteorological records from global weather station networks, satellite observations, and paleoclimate reconstructions. Climate model experiments include both historical simulations and counterfactual scenarios with and without anthropogenic forcing to isolate human influences.

                Results demonstrate clear anthropogenic fingerprints in multiple categories of extreme events. Heat extremes show the strongest attribution signals, with anthropogenic climate change increasing the likelihood of severe heatwaves by factors of 10-100 in many regions. Precipitation extremes show more complex patterns but clear trends toward more intense events in many areas.

                Regional analysis reveals significant geographical variations in attribution strength and event characteristics. Mediterranean regions show particularly strong signals for drought attribution, while Arctic areas demonstrate clear human influence on temperature extremes and ice-related events.

                Key findings include: (1) Quantitative attribution of major recent extreme events to anthropogenic climate change, (2) Regional and seasonal patterns in attribution strength across different event types, (3) Improved understanding of the physical mechanisms linking greenhouse gas emissions to extreme weather, (4) Enhanced risk assessment capabilities for climate adaptation planning.

                The research provides crucial input for climate risk assessment, insurance industry applications, and legal frameworks addressing climate damages. Our attribution methodologies enable quantitative assessment of climate change contributions to specific damaging events.

                Uncertainty quantification is a critical component of the analysis, with comprehensive assessment of methodological uncertainties, model limitations, and observational constraints. We develop robust uncertainty bounds that account for multiple sources of uncertainty in the attribution process.

                Impacts on human systems are documented through case studies linking attributed extreme events to economic damages, human health impacts, and ecosystem disruptions. These connections provide essential information for climate adaptation and disaster risk reduction strategies.

                The findings contribute to international climate policy discussions and legal frameworks addressing loss and damage from climate change. Our work provides scientific foundation for climate litigation and compensation mechanisms.

                Future research will focus on real-time attribution capabilities, improved understanding of compound events, and enhanced regional attribution methodologies for data-sparse regions. The development of operational attribution systems will enable rapid assessment of climate change contributions to ongoing extreme events.
                """
            }
        ],
        
        ScientificCategory.BIOTECHNOLOGY: [
            {
                "title": "CRISPR-Cas9 Applications in Gene Therapy: Recent Advances and Clinical Prospects",
                "abstract": "Comprehensive review of CRISPR-Cas9 gene editing technology applications in clinical gene therapy, highlighting recent advances, successful treatments, and ongoing challenges in translating research to clinical practice.",
                "authors": ["Dr. Elena Genetica", "Prof. Carlos Biotech", "Dr. Edição Genômica"],
                "keywords": ["CRISPR", "gene therapy", "biotechnology", "genetic engineering", "clinical applications"],
                "content": """
                CRISPR-Cas9 technology has emerged as a revolutionary tool in biotechnology, offering unprecedented precision in gene editing with transformative implications for therapeutic applications. This comprehensive review examines recent advances and clinical prospects of CRISPR-based gene therapy, analyzing both successes and ongoing challenges in translating laboratory discoveries to clinical practice.

                The methodological foundation covers the fundamental mechanisms of CRISPR-Cas9 systems, including guide RNA design principles, target recognition specificity, and DNA repair pathway utilization. We analyze recent technological improvements in editing efficiency, specificity enhancement, and delivery system optimization that have significantly advanced the therapeutic potential of this platform.

                Clinical application analysis encompasses multiple therapeutic areas including inherited genetic disorders, acquired diseases, and cancer treatment. We review completed and ongoing clinical trials, analyzing treatment outcomes, safety profiles, and efficacy measurements across different patient populations and disease conditions.

                Recent clinical trials have demonstrated remarkable success in treating previously incurable genetic disorders. Notable achievements include successful treatments for sickle cell disease and β-thalassemia through ex vivo editing of patient hematopoietic stem cells, with patients achieving transfusion independence and improved quality of life.

                Cancer immunotherapy applications represent another major success area, with CAR-T cell therapies enhanced through CRISPR editing showing improved efficacy and reduced side effects. The technology enables precise modification of T cells to enhance their tumor-targeting capabilities while reducing off-target effects.

                Key technological advances include: (1) Improved guide RNA design algorithms reducing off-target effects by over 95%, (2) Enhanced delivery systems including lipid nanoparticles and adeno-associated virus vectors, (3) Base editing and prime editing technologies enabling precise nucleotide changes without double-strand breaks, (4) Multiplexed editing capabilities allowing simultaneous modification of multiple genes.

                Delivery system development has overcome significant barriers to in vivo gene editing, with new formulations achieving tissue-specific targeting and improved cellular uptake. Lipid nanoparticle systems show particular promise for liver-directed therapies, while viral vectors enable targeting of specific cell types including neurons and muscle cells.

                Safety considerations remain paramount, with comprehensive analysis of off-target effects, immune responses, and long-term consequences. Recent studies demonstrate significantly improved safety profiles through enhanced specificity and better understanding of cellular DNA repair mechanisms.

                Regulatory frameworks are evolving to accommodate CRISPR therapeutics, with agencies developing specific guidelines for gene editing clinical trials. The approval of CTX001 for sickle cell disease and β-thalassemia represents a milestone in regulatory acceptance of CRISPR therapies.

                Economic analysis reveals the potential for CRISPR therapies to provide cost-effective treatments for previously untreatable conditions, despite high initial development costs. The one-time treatment potential of many gene editing approaches offers significant advantages over chronic treatment regimens.

                Ethical considerations encompass both clinical applications and broader societal implications of human genetic modification. Current clinical applications focus on somatic cell editing, avoiding heritable changes while addressing serious medical conditions.

                Future developments focus on expanding the range of treatable conditions, improving delivery efficiency, and reducing treatment costs. In vivo applications are advancing rapidly, with promising results for treating genetic blindness, muscular dystrophy, and other conditions requiring direct tissue targeting.

                The field is rapidly evolving toward more sophisticated applications including tissue regeneration, organ engineering, and complex genetic disease treatment. These advances promise to transform medicine by providing curative treatments for conditions that currently have limited therapeutic options.
                """
            },
            {
                "title": "Synthetic Biology Approaches for Sustainable Biofuel Production",
                "abstract": "Novel synthetic biology strategies for engineering microorganisms to produce sustainable biofuels, focusing on metabolic pathway optimization and industrial-scale production challenges.",
                "authors": ["Prof. Sintetica Biologia", "Dr. Combustivel Verde"],
                "keywords": ["synthetic biology", "biofuels", "metabolic engineering", "sustainability", "bioprocessing"],
                "content": """
                Synthetic biology approaches offer transformative potential for sustainable biofuel production by enabling the engineering of microorganisms to efficiently convert renewable feedstocks into transportation fuels. This research presents novel strategies for metabolic pathway optimization and addresses key challenges in scaling synthetic biology solutions for industrial biofuel production.

                Our methodological framework integrates computational pathway design, genetic circuit engineering, and systems-level optimization to create microbial cell factories capable of high-efficiency fuel production. We employ advanced DNA assembly techniques, including modular cloning systems and automated strain construction pipelines, to rapidly build and test engineered organisms.

                The metabolic engineering strategy focuses on optimizing native and heterologous pathways for biofuel precursor synthesis, including fatty acid derivatives, terpenoids, and advanced alcohols. We implement dynamic pathway regulation using synthetic circuits that respond to cellular metabolic states, maximizing fuel production while maintaining cellular viability.

                Strain development encompasses multiple microbial platforms including Escherichia coli, Saccharomyces cerevisiae, and oleaginous yeasts, each optimized for specific fuel molecules and feedstock utilization. Comparative analysis reveals distinct advantages for different organism-fuel combinations based on metabolic capacity and industrial robustness.

                Results demonstrate significant improvements in fuel production titers, yields, and productivities compared to previous approaches. Engineered E. coli strains achieve fatty acid ethyl ester production rates exceeding 2 g/L/h with yields approaching theoretical maximums. Yeast-based systems show superior performance for complex molecule production and industrial stress tolerance.

                Key innovations include: (1) Dynamic metabolic control circuits that optimize pathway flux in response to cellular conditions, (2) Modular pathway architectures enabling rapid strain optimization and fuel molecule diversification, (3) Advanced fermentation strategies integrating continuous processing with real-time metabolic monitoring, (4) Sustainable feedstock utilization including agricultural residues and waste streams.

                Bioprocessing optimization addresses critical scale-up challenges including oxygen transfer, heat management, and product recovery. We develop fed-batch and continuous fermentation strategies that maintain high productivity while minimizing production costs and environmental impact.

                Economic analysis demonstrates the potential for cost-competitive biofuel production through optimized strain performance and process integration. Life cycle assessment reveals significant environmental benefits compared to petroleum-derived fuels, including reduced greenhouse gas emissions and improved sustainability profiles.

                Industrial implementation challenges are addressed through partnerships with biotechnology companies and pilot-scale demonstration projects. These collaborations provide essential insights into real-world production constraints and market requirements for sustainable biofuel technologies.

                Product diversification strategies enable the production of multiple fuel molecules and high-value co-products from the same microbial platform, improving overall process economics. Integration with existing petroleum refining infrastructure facilitates market adoption and reduces implementation barriers.

                Safety and containment considerations are paramount in industrial biotechnology applications, with comprehensive risk assessment and mitigation strategies for engineered organism deployment. Biological containment systems prevent environmental release while maintaining industrial productivity.

                Regulatory frameworks for synthetic biology applications in biofuel production are evolving, with industry engagement helping to shape appropriate oversight mechanisms that ensure safety while enabling innovation.

                Future research directions include expanding feedstock utilization to include CO2 and other waste streams, developing more efficient metabolic pathways through protein engineering, and creating integrated biorefineries that maximize resource utilization and minimize waste production.
                """
            },
            {
                "title": "Personalized Medicine Through Genomic Data Analysis and AI Integration",
                "abstract": "Integration of genomic data analysis with artificial intelligence to enable personalized medicine approaches, focusing on drug response prediction and disease risk assessment for individual patients.",
                "authors": ["Dr. Personalizada Medicina", "Prof. Genoma Inteligente"],
                "keywords": ["personalized medicine", "genomics", "artificial intelligence", "pharmacogenomics", "precision healthcare"],
                "content": """
                The integration of genomic data analysis with artificial intelligence represents a paradigm shift toward truly personalized medicine, enabling healthcare providers to tailor treatments based on individual genetic profiles, lifestyle factors, and environmental influences. This research develops comprehensive frameworks for implementing AI-driven personalized medicine approaches in clinical practice.

                Our methodological approach combines large-scale genomic data analysis, machine learning algorithms, and clinical informatics to create predictive models for drug response, disease susceptibility, and treatment optimization. We utilize multi-omics data integration including genomics, transcriptomics, proteomics, and metabolomics to build comprehensive patient profiles.

                The AI framework incorporates deep learning architectures specifically designed for genomic data analysis, including convolutional neural networks for sequence analysis and graph neural networks for pathway modeling. These models are trained on diverse population datasets to ensure broad applicability and reduce bias in clinical applications.

                Pharmacogenomic analysis focuses on predicting individual drug responses based on genetic variations in drug metabolism, transport, and target genes. We develop algorithms that integrate pharmacokinetic modeling with genetic data to optimize drug dosing and selection for individual patients, significantly reducing adverse drug reactions and improving therapeutic outcomes.

                Clinical validation involves retrospective analysis of electronic health records combined with prospective clinical trials to validate AI predictions in real-world healthcare settings. Collaboration with healthcare institutions provides access to diverse patient populations and clinical outcomes data essential for model development and validation.

                Results demonstrate significant improvements in treatment outcomes through personalized medicine approaches. Drug response prediction models achieve 85% accuracy in identifying patients likely to experience adverse reactions, enabling proactive treatment modifications. Disease risk assessment models successfully stratify patients into appropriate screening and prevention programs.

                Key developments include: (1) Multi-modal AI architectures that integrate diverse biological data types for comprehensive patient profiling, (2) Real-time clinical decision support systems that provide actionable recommendations during patient encounters, (3) Population-specific models that account for genetic diversity and health disparities across different ethnic groups, (4) Privacy-preserving algorithms that enable collaborative research while protecting patient data.

                Clinical implementation strategies address practical challenges including electronic health record integration, physician training, and workflow optimization. User interface design focuses on presenting complex genomic information in clinically actionable formats that support rapid decision-making in healthcare settings.

                Ethical considerations encompass patient consent for genomic data use, data privacy protection, and equitable access to personalized medicine technologies. We develop frameworks for responsible AI deployment that prioritize patient autonomy and minimize potential for discrimination based on genetic information.

                Cost-effectiveness analysis demonstrates the economic benefits of personalized medicine through reduced adverse drug reactions, improved treatment efficacy, and prevention of costly disease complications. These benefits justify the initial investment in genomic testing and AI infrastructure development.

                Regulatory compliance addresses evolving requirements for AI-based medical devices and genomic testing in clinical practice. Collaboration with regulatory agencies helps establish appropriate validation standards and approval pathways for personalized medicine technologies.

                Health disparities research investigates the potential for AI-driven personalized medicine to either exacerbate or reduce existing healthcare inequalities. We develop strategies for ensuring equitable access to personalized medicine benefits across diverse populations and socioeconomic groups.

                Technology transfer initiatives facilitate the translation of research discoveries into clinical products and services, with partnerships between academic institutions, biotechnology companies, and healthcare providers driving innovation and implementation.

                Future directions include expanding AI capabilities to incorporate real-time physiological monitoring data, developing personalized treatment strategies for complex diseases like cancer and neurological disorders, and creating population health management tools that optimize healthcare resource allocation based on individual risk profiles.
                """
            }
        ]
    }
    
    # Add articles to vector store
    total_added = 0
    for category, articles in reference_articles.items():
        print(f"\n📂 Adding {category.value} articles...")
        
        for i, article_data in enumerate(articles, 1):
            article = ArticleContent(
                title=article_data["title"],
                abstract=article_data["abstract"],
                authors=article_data["authors"],
                keywords=article_data["keywords"],
                full_text=article_data["content"]
            )
            
            success = await vector_store.add_document(article, category)
            if success:
                total_added += 1
                print(f"  ✅ Article {i}: {article.title[:60]}...")
            else:
                print(f"  ❌ Failed to add article {i}")
    
    # Get final statistics
    stats = vector_store.get_statistics()
    
    print(f"\n📊 Vector Store Statistics:")
    print(f"  - Total documents: {stats['total_documents']}")
    print(f"  - Total chunks: {stats['total_chunks']}")
    print(f"  - Embedding dimensions: {stats['embedding_dimension']}")
    print(f"  - FAISS enabled: {stats['faiss_available']}")
    print(f"  - Storage path: {stats['storage_path']}")
    
    print(f"\n✅ Vector store setup completed! Added {total_added} articles with {stats['total_chunks']} chunks.")
    
    return vector_store

if __name__ == "__main__":
    asyncio.run(setup_vector_store())