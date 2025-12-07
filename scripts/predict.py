#!/usr/bin/env python
"""
Inference script for ADMET predictions.

Usage:
    python scripts/predict.py --smiles "CCO" --property Caco2_Wang
    python scripts/predict.py --smiles "CCO" --property Caco2_Wang --explain
    python scripts/predict.py --input molecules.csv --output predictions.csv
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from torch_geometric.data import Batch

from src.data import MolecularFeaturizer
from src.models import GNNPredictor
from src.rag import VectorStore, ADMETExplainer
from src.utils import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ADMET predictions"
    )
    parser.add_argument(
        "--smiles",
        type=str,
        help="Single SMILES string to predict"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="CSV file with SMILES column"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file for batch predictions"
    )
    parser.add_argument(
        "--property",
        type=str,
        default="Caco2_Wang",
        help="ADMET property to predict"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate RAG-based explanation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint"
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict, input_dim: int, edge_dim: int):
    """Load trained model from checkpoint."""
    model_config = config["model"]
    
    model = GNNPredictor(
        input_dim=input_dim,
        hidden_dim=model_config.get("hidden_dim", 128),
        output_dim=model_config.get("output_dim", 1),
        num_layers=model_config.get("num_layers", 3),
        dropout=model_config.get("dropout", 0.2),
        architecture=model_config.get("architecture", "GCN"),
        pooling=model_config.get("pooling", "mean"),
        edge_dim=edge_dim
    )
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


def predict_single(
    smiles: str,
    model,
    featurizer: MolecularFeaturizer,
    explainer=None,
    property_name: str = "Caco2_Wang"
) -> dict:
    """Generate prediction for a single molecule."""
    graph = featurizer.smiles_to_graph(smiles)
    if graph is None:
        return {"error": f"Failed to parse SMILES: {smiles}"}
    
    descriptors = featurizer.get_molecular_descriptors(smiles)
    
    batch = Batch.from_data_list([graph])
    
    with torch.no_grad():
        prediction = model(batch).item()
    
    result = {
        "smiles": smiles,
        "property": property_name,
        "prediction": prediction,
        "molecular_weight": descriptors["molecular_weight"],
        "logp": descriptors["logp"],
        "tpsa": descriptors["tpsa"],
        "h_donors": descriptors["num_h_donors"],
        "h_acceptors": descriptors["num_h_acceptors"],
        "rotatable_bonds": descriptors["num_rotatable_bonds"],
    }
    
    if explainer is not None:
        explanation, context = explainer.explain(
            smiles=smiles,
            property_name=property_name,
            prediction=prediction,
            molecular_descriptors=descriptors
        )
        result["explanation"] = explanation
    
    return result


def main():
    """Main prediction function."""
    args = parse_args()
    
    if not args.smiles and not args.input:
        print("Error: Provide either --smiles or --input")
        sys.exit(1)
    
    config = load_config(args.config)
    
    print("Loading featurizer...")
    featurizer = MolecularFeaturizer()
    
    print("Loading model...")
    model = load_model(
        args.checkpoint,
        config,
        input_dim=featurizer.atom_feature_dim,
        edge_dim=featurizer.bond_feature_dim
    )
    
    explainer = None
    if args.explain:
        print("Loading RAG components...")
        rag_config = config.get("rag", {})
        
        vector_store = VectorStore(
            collection_name=rag_config.get("collection_name", "admet_knowledge"),
            embedding_model=rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
            persist_directory=rag_config.get("persist_directory", "./chroma_db")
        )
        
        explainer = ADMETExplainer(
            vector_store=vector_store,
            llm_model=rag_config.get("llm_model", "gpt-4o-mini"),
            temperature=rag_config.get("temperature", 0.3)
        )
    
    if args.smiles:
        print(f"\nPredicting for: {args.smiles}")
        print("-" * 60)
        
        result = predict_single(
            smiles=args.smiles,
            model=model,
            featurizer=featurizer,
            explainer=explainer,
            property_name=args.property
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Property: {result['property']}")
            print(f"Prediction: {result['prediction']:.4f}")
            print(f"\nMolecular Descriptors:")
            print(f"  MW: {result['molecular_weight']:.1f} Da")
            print(f"  LogP: {result['logp']:.2f}")
            print(f"  TPSA: {result['tpsa']:.1f} A^2")
            print(f"  H-Donors: {result['h_donors']}")
            print(f"  H-Acceptors: {result['h_acceptors']}")
            print(f"  Rotatable Bonds: {result['rotatable_bonds']}")
            
            if "explanation" in result:
                print(f"\nExplanation:")
                print(result["explanation"])
    
    if args.input:
        print(f"\nProcessing batch from: {args.input}")
        
        df = pd.read_csv(args.input)
        smiles_col = None
        for col in ["SMILES", "smiles", "Smiles", "Drug"]:
            if col in df.columns:
                smiles_col = col
                break
        
        if smiles_col is None:
            print("Error: No SMILES column found in input file")
            sys.exit(1)
        
        results = []
        for idx, row in df.iterrows():
            smiles = row[smiles_col]
            result = predict_single(
                smiles=smiles,
                model=model,
                featurizer=featurizer,
                explainer=explainer,
                property_name=args.property
            )
            results.append(result)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} molecules")
        
        results_df = pd.DataFrame(results)
        
        output_path = args.output or args.input.replace(".csv", "_predictions.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved predictions to: {output_path}")


if __name__ == "__main__":
    main()
