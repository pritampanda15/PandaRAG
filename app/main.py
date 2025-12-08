"""
Streamlit application for ADMET property prediction.

Launch with: streamlit run app/main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
from torch_geometric.data import Batch
import pandas as pd
import numpy as np

from src.data import MolecularFeaturizer
from src.models import GNNPredictor
from src.rag import VectorStore, ADMETExplainer
from src.utils import load_config


st.set_page_config(
    page_title="ADMET Predictor",
    page_icon="🧪",
    layout="wide"
)


@st.cache_resource
def load_resources():
    """Load model and RAG components."""
    config = load_config("config/config.yaml")
    
    featurizer = MolecularFeaturizer()
    
    model_config = config["model"]
    model = GNNPredictor(
        input_dim=featurizer.atom_feature_dim,
        hidden_dim=model_config.get("hidden_dim", 128),
        output_dim=model_config.get("output_dim", 1),
        num_layers=model_config.get("num_layers", 3),
        dropout=model_config.get("dropout", 0.2),
        architecture=model_config.get("architecture", "GCN"),
        pooling=model_config.get("pooling", "mean"),
        edge_dim=featurizer.bond_feature_dim
    )
    
    checkpoint_path = "models/best_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    rag_config = config.get("rag", {})
    vector_store = VectorStore(
        collection_name=rag_config.get("collection_name", "admet_knowledge"),
        embedding_model=rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
        persist_directory=rag_config.get("persist_directory", "./chroma_db")
    )
    
    explainer = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            explainer = ADMETExplainer(
                vector_store=vector_store,
                llm_model=rag_config.get("llm_model", "gpt-4o-mini")
            )
        except Exception as e:
            st.warning(f"RAG explainer not available: {e}")
    
    return config, featurizer, model, vector_store, explainer


PROPERTY_INFO = {
    "Caco2_Wang": {
        "name": "Caco-2 Permeability",
        "description": "Intestinal absorption potential",
        "unit": "log Papp (cm/s)",
        "type": "regression"
    },
    "HIA_Hou": {
        "name": "Human Intestinal Absorption",
        "description": "Percentage absorbed through intestine",
        "unit": "probability",
        "type": "classification"
    },
    "BBB_Martins": {
        "name": "Blood-Brain Barrier",
        "description": "CNS penetration potential",
        "unit": "probability",
        "type": "classification"
    },
    "hERG": {
        "name": "hERG Inhibition",
        "description": "Cardiac toxicity risk",
        "unit": "probability",
        "type": "classification"
    },
    "AMES": {
        "name": "AMES Mutagenicity",
        "description": "Mutagenic potential",
        "unit": "probability",
        "type": "classification"
    },
}


def predict_molecule(smiles: str, model, featurizer):
    """Generate prediction for a molecule."""
    graph = featurizer.smiles_to_graph(smiles)
    if graph is None:
        return None, None
    
    descriptors = featurizer.get_molecular_descriptors(smiles)
    batch = Batch.from_data_list([graph])
    
    with torch.no_grad():
        prediction = model(batch).item()
    
    return prediction, descriptors


def main():
    """Main application."""
    st.title("ADMET Property Prediction")
    st.markdown("Predict pharmacokinetic properties from molecular structure")
    
    config, featurizer, model, vector_store, explainer = load_resources()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_input(
            "Enter SMILES",
            value="CC(=O)OC1=CC=CC=C1C(=O)O",
            help="SMILES representation of the molecule"
        )
    
    with col2:
        property_name = st.selectbox(
            "ADMET Property",
            options=list(PROPERTY_INFO.keys()),
            format_func=lambda x: PROPERTY_INFO[x]["name"]
        )
    
    generate_explanation = st.checkbox(
        "Generate AI explanation",
        value=explainer is not None,
        disabled=explainer is None
    )
    
    if st.button("Predict", type="primary"):
        if not smiles_input.strip():
            st.error("Please enter a SMILES string")
            return
        
        with st.spinner("Generating prediction..."):
            prediction, descriptors = predict_molecule(
                smiles_input, model, featurizer
            )
        
        if prediction is None:
            st.error("Failed to parse SMILES. Please check the input.")
            return
        
        prop_info = PROPERTY_INFO[property_name]
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=prop_info["name"],
                value=f"{prediction:.4f}",
                help=prop_info["description"]
            )
        
        with col2:
            if prop_info["type"] == "classification":
                prob = 1 / (1 + np.exp(-prediction))
                status = "Positive" if prob > 0.5 else "Negative"
                st.metric("Classification", status)
        
        with col3:
            st.metric("Unit", prop_info["unit"])
        
        st.markdown("### Molecular Descriptors")
        
        desc_col1, desc_col2, desc_col3 = st.columns(3)
        
        with desc_col1:
            st.metric("Molecular Weight", f"{descriptors['molecular_weight']:.1f} Da")
            st.metric("LogP", f"{descriptors['logp']:.2f}")
        
        with desc_col2:
            st.metric("TPSA", f"{descriptors['tpsa']:.1f} A^2")
            st.metric("Rotatable Bonds", descriptors["num_rotatable_bonds"])
        
        with desc_col3:
            st.metric("H-Bond Donors", descriptors["num_h_donors"])
            st.metric("H-Bond Acceptors", descriptors["num_h_acceptors"])
        
        if generate_explanation and explainer is not None:
            st.markdown("### AI Explanation")
            
            with st.spinner("Generating explanation..."):
                try:
                    explanation, context = explainer.explain(
                        smiles=smiles_input,
                        property_name=property_name,
                        prediction=prediction,
                        molecular_descriptors=descriptors
                    )
                    st.markdown(explanation)
                    
                    with st.expander("Retrieved Knowledge Sources"):
                        st.text(context)
                        
                except Exception as e:
                    st.error(f"Failed to generate explanation: {e}")
    
    st.markdown("---")
    st.markdown("### Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV with SMILES column",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        smiles_col = None
        for col in ["SMILES", "smiles", "Smiles", "Drug"]:
            if col in df.columns:
                smiles_col = col
                break
        
        if smiles_col is None:
            st.error("No SMILES column found")
        else:
            st.write(f"Found {len(df)} molecules")
            
            if st.button("Run Batch Prediction"):
                progress = st.progress(0)
                results = []
                
                for idx, row in df.iterrows():
                    smiles = row[smiles_col]
                    pred, desc = predict_molecule(smiles, model, featurizer)
                    
                    results.append({
                        "SMILES": smiles,
                        "Prediction": pred if pred is not None else "Error",
                        "MW": desc["molecular_weight"] if desc else None,
                        "LogP": desc["logp"] if desc else None,
                    })
                    
                    progress.progress((idx + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )


if __name__ == "__main__":
    main()
