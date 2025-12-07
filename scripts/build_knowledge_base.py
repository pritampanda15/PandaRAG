#!/usr/bin/env python
"""
Build the ADMET knowledge base for RAG.

This script processes documents and creates the vector store
for retrieving pharmacokinetic knowledge.

Usage:
    python scripts/build_knowledge_base.py --docs-path data/knowledge_base/
    python scripts/build_knowledge_base.py --seed-defaults
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import DocumentProcessor, VectorStore
from src.utils import load_config


ADMET_SEED_DOCUMENTS = [
    {
        "property_name": "Caco2_Wang",
        "description": "Caco-2 cells are human colon adenocarcinoma cells used as an in vitro model for intestinal drug absorption. The assay measures apparent permeability (Papp) which correlates with human intestinal absorption.",
        "interpretation_guide": "Log Papp values above -5.15 cm/s indicate good permeability. High lipophilicity (LogP) and low polar surface area (TPSA < 140 A^2) favor absorption. Molecular weight above 500 Da typically reduces permeability.",
        "thresholds": {"good_permeability": -5.15, "poor_permeability": -6.0},
        "references": [
            "Artursson P, Karlsson J. Correlation between oral drug absorption in humans and apparent drug permeability coefficients in human intestinal epithelial (Caco-2) cells. Biochem Biophys Res Commun. 1991.",
            "Wang NN, et al. ADME Properties Evaluation in Drug Discovery. Acta Pharmaceutica Sinica B. 2016."
        ]
    },
    {
        "property_name": "HIA_Hou",
        "description": "Human Intestinal Absorption (HIA) measures the percentage of an orally administered drug absorbed through the intestinal wall. It is a critical parameter for oral bioavailability.",
        "interpretation_guide": "HIA+ means greater than 30% absorption. Key factors include lipophilicity, hydrogen bonding capacity, molecular size, and ionization state at intestinal pH. Lipinski Rule of Five violations often correlate with poor HIA.",
        "thresholds": {"high_absorption": 0.3, "classification_threshold": 0.5},
        "references": [
            "Hou T, et al. ADME evaluation in drug discovery. J Chem Inf Model. 2007.",
            "Lipinski CA. Lead- and drug-like compounds: the rule-of-five revolution. Drug Discov Today Technol. 2004."
        ]
    },
    {
        "property_name": "BBB_Martins",
        "description": "Blood-Brain Barrier (BBB) penetration indicates whether a compound can cross from blood into the central nervous system. This is essential for CNS drugs but may be undesirable for peripherally acting drugs.",
        "interpretation_guide": "BBB+ compounds are CNS-active. Factors favoring BBB penetration include low molecular weight (<450 Da), moderate lipophilicity (LogP 1-3), low TPSA (<90 A^2), and few hydrogen bond donors (<3). P-glycoprotein substrate status also affects brain exposure.",
        "thresholds": {"classification_threshold": 0.5},
        "references": [
            "Martins IF, et al. A Bayesian approach to in silico blood-brain barrier penetration modeling. J Chem Inf Model. 2012.",
            "Pardridge WM. Blood-brain barrier drug targeting. Drug Discov Today. 2007."
        ]
    },
    {
        "property_name": "PPBR_AZ",
        "description": "Plasma Protein Binding Rate (PPBR) measures the fraction of drug bound to plasma proteins, primarily albumin and alpha-1 acid glycoprotein. Only unbound drug is pharmacologically active and available for distribution.",
        "interpretation_guide": "High binding (>90%) reduces free drug concentration and may prolong half-life. Binding above 99% is concerning as small changes significantly affect free fraction. Drug-drug interactions can occur through binding displacement.",
        "thresholds": {"high_binding": 90.0, "very_high_binding": 99.0},
        "references": [
            "Smith DA, et al. The effect of plasma protein binding on in vivo efficacy. Drug Discov Today. 2010.",
            "Trainor GL. The importance of plasma protein binding in drug discovery. Expert Opin Drug Discov. 2007."
        ]
    },
    {
        "property_name": "CYP2D6_Veith",
        "description": "CYP2D6 is a cytochrome P450 enzyme responsible for metabolizing approximately 25% of clinically used drugs. Inhibition can cause drug-drug interactions by altering the metabolism of co-administered CYP2D6 substrates.",
        "interpretation_guide": "CYP2D6 inhibitors may increase plasma concentrations of CYP2D6 substrates like codeine, tamoxifen, and many antidepressants. Basic nitrogen atoms and aromatic rings are common features of CYP2D6 inhibitors.",
        "thresholds": {"classification_threshold": 0.5},
        "references": [
            "Veith H, et al. Comprehensive characterization of cytochrome P450 isozyme selectivity. Nat Biotechnol. 2009.",
            "Zhou SF. Polymorphism of human cytochrome P450 2D6. Clin Pharmacokinet. 2009."
        ]
    },
    {
        "property_name": "hERG",
        "description": "hERG (human Ether-a-go-go Related Gene) encodes a cardiac potassium channel. Drug-induced hERG inhibition can cause QT prolongation and potentially fatal cardiac arrhythmias (Torsades de Pointes).",
        "interpretation_guide": "hERG inhibition is a major cause of drug withdrawals. Structural alerts include basic amines, aromatic rings, and lipophilic groups. IC50 below 10 uM is concerning. The safety margin between therapeutic plasma concentration and hERG IC50 should be at least 30-fold.",
        "thresholds": {"classification_threshold": 0.5},
        "references": [
            "Sanguinetti MC, Tristani-Firouzi M. hERG potassium channels and cardiac arrhythmia. Nature. 2006.",
            "Redfern WS, et al. Relationships between preclinical cardiac electrophysiology, clinical QT interval prolongation and torsade de pointes. Br J Pharmacol. 2003."
        ]
    },
    {
        "property_name": "AMES",
        "description": "The Ames test is a bacterial reverse mutation assay used to identify potential mutagens. A positive result indicates the compound can cause DNA damage and may be carcinogenic.",
        "interpretation_guide": "AMES+ compounds carry mutagenic risk and typically cannot proceed in drug development without strong justification. Structural alerts include aromatic amines, nitro groups, alkylating agents, and polycyclic aromatic hydrocarbons.",
        "thresholds": {"classification_threshold": 0.5},
        "references": [
            "Ames BN, et al. Methods for detecting carcinogens and mutagens with the Salmonella/mammalian-microsome mutagenicity test. Mutat Res. 1975.",
            "Kazius J, et al. Derivation and validation of toxicophores for mutagenicity prediction. J Med Chem. 2005."
        ]
    },
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build ADMET knowledge base"
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        default="data/knowledge_base/",
        help="Path to document directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--seed-defaults",
        action="store_true",
        help="Seed with default ADMET property documents"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before building"
    )
    return parser.parse_args()


def main():
    """Main knowledge base building function."""
    args = parse_args()
    
    config = load_config(args.config)
    rag_config = config.get("rag", {})
    
    print("=" * 60)
    print("ADMET Knowledge Base Builder")
    print("=" * 60)
    
    vector_store = VectorStore(
        collection_name=rag_config.get("collection_name", "admet_knowledge"),
        embedding_model=rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
        persist_directory=rag_config.get("persist_directory", "./chroma_db")
    )
    
    if args.reset:
        print("\nResetting collection...")
        vector_store.delete_collection()
    
    processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    if args.seed_defaults:
        print("\nSeeding default ADMET property documents...")
        
        seed_docs = []
        for doc_data in ADMET_SEED_DOCUMENTS:
            doc = processor.create_admet_document(**doc_data)
            seed_docs.append(doc)
        
        added = vector_store.add_documents(seed_docs)
        print(f"Added {added} seed documents")
    
    if os.path.exists(args.docs_path):
        print(f"\nProcessing documents from: {args.docs_path}")
        
        documents = list(processor.process_directory(args.docs_path))
        
        if documents:
            added = vector_store.add_documents(documents)
            print(f"Added {added} document chunks")
        else:
            print("No documents found to process")
    else:
        print(f"\nDocument path not found: {args.docs_path}")
        print("Skipping document processing")
    
    print("\n" + "=" * 60)
    print("Knowledge Base Statistics")
    print("=" * 60)
    
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTesting retrieval...")
    results = vector_store.search("caco-2 permeability absorption", top_k=3)
    print(f"Found {len(results)} results for test query")
    
    if results:
        print("\nTop result preview:")
        content, metadata, score = results[0]
        print(f"  Score: {score:.3f}")
        print(f"  Source: {metadata.get('source', 'unknown')}")
        print(f"  Content: {content[:200]}...")
    
    print("\nKnowledge base ready.")


if __name__ == "__main__":
    main()
