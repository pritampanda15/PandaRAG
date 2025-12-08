"""
LLM-based explanation generation for ADMET predictions.

Uses retrieved context to generate evidence-based interpretations
of molecular property predictions.
"""

import os
from typing import Dict, Optional, Tuple
import httpx
import certifi
from openai import OpenAI
from dotenv import load_dotenv

from .vector_store import VectorStore


load_dotenv()


class ADMETExplainer:
    """
    Generates explanations for ADMET predictions using RAG.
    
    Combines GNN predictions with retrieved pharmacokinetic
    knowledge to produce interpretable explanations.
    
    Attributes:
        vector_store: VectorStore for knowledge retrieval.
        llm_model: OpenAI model identifier.
    """
    
    PROPERTY_DESCRIPTIONS = {
        "Caco2_Wang": {
            "name": "Caco-2 Permeability",
            "unit": "log Papp (cm/s)",
            "interpretation": "Higher values indicate better intestinal absorption",
            "threshold": -5.15,
            "threshold_meaning": "Values > -5.15 suggest good permeability"
        },
        "HIA_Hou": {
            "name": "Human Intestinal Absorption",
            "unit": "binary (0/1)",
            "interpretation": "1 indicates high absorption (>30%)",
            "threshold": 0.5,
            "threshold_meaning": "Probability > 0.5 classified as absorbed"
        },
        "BBB_Martins": {
            "name": "Blood-Brain Barrier Penetration",
            "unit": "binary (0/1)",
            "interpretation": "1 indicates BBB permeable",
            "threshold": 0.5,
            "threshold_meaning": "Probability > 0.5 classified as BBB+"
        },
        "PPBR_AZ": {
            "name": "Plasma Protein Binding Rate",
            "unit": "percentage",
            "interpretation": "Higher binding reduces free drug concentration",
            "threshold": 90,
            "threshold_meaning": "Values > 90% indicate high binding"
        },
        "CYP2D6_Veith": {
            "name": "CYP2D6 Inhibition",
            "unit": "binary (0/1)",
            "interpretation": "1 indicates inhibition risk",
            "threshold": 0.5,
            "threshold_meaning": "Probability > 0.5 suggests inhibition"
        },
        "hERG": {
            "name": "hERG Channel Inhibition",
            "unit": "binary (0/1)",
            "interpretation": "1 indicates cardiotoxicity risk",
            "threshold": 0.5,
            "threshold_meaning": "Probability > 0.5 suggests hERG liability"
        },
        "AMES": {
            "name": "AMES Mutagenicity",
            "unit": "binary (0/1)",
            "interpretation": "1 indicates mutagenic potential",
            "threshold": 0.5,
            "threshold_meaning": "Probability > 0.5 classified as mutagenic"
        },
    }
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        """
        Initialize explainer.
        
        Args:
            vector_store: Configured VectorStore instance.
            llm_model: OpenAI model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
        """
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Create HTTP client with explicit SSL certificate configuration
        http_client = httpx.Client(
            verify=certifi.where(),
            timeout=30.0
        )

        self.client = OpenAI(api_key=api_key, http_client=http_client)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the LLM."""
        return """You are a pharmacokinetics expert assistant. Your role is to 
explain ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) 
predictions for drug molecules.

When explaining predictions:
1. Interpret the predicted value in context of the property
2. Relate molecular features to the prediction when relevant
3. Cite retrieved knowledge to support your explanation
4. Mention any limitations or caveats
5. Keep explanations concise and scientifically accurate

Do not make up information. If the context does not support a claim, say so."""
    
    def _build_user_prompt(
        self,
        smiles: str,
        property_name: str,
        prediction: float,
        molecular_descriptors: Dict[str, float],
        context: str
    ) -> str:
        """Build user prompt with prediction details and context."""
        prop_info = self.PROPERTY_DESCRIPTIONS.get(property_name, {})
        
        prompt = f"""Explain the following ADMET prediction:

MOLECULE: {smiles}

PROPERTY: {prop_info.get('name', property_name)}
PREDICTED VALUE: {prediction:.4f} {prop_info.get('unit', '')}
THRESHOLD: {prop_info.get('threshold_meaning', 'N/A')}

MOLECULAR FEATURES:
- Molecular Weight: {molecular_descriptors.get('molecular_weight', 'N/A'):.1f} Da
- LogP: {molecular_descriptors.get('logp', 'N/A'):.2f}
- TPSA: {molecular_descriptors.get('tpsa', 'N/A'):.1f} A^2
- H-Bond Donors: {molecular_descriptors.get('num_h_donors', 'N/A')}
- H-Bond Acceptors: {molecular_descriptors.get('num_h_acceptors', 'N/A')}
- Rotatable Bonds: {molecular_descriptors.get('num_rotatable_bonds', 'N/A')}

RETRIEVED KNOWLEDGE:
{context}

Provide a concise explanation of what this prediction means for the molecule's 
pharmacokinetic profile. Reference the molecular features and retrieved 
knowledge where appropriate."""
        
        return prompt
    
    def explain(
        self,
        smiles: str,
        property_name: str,
        prediction: float,
        molecular_descriptors: Dict[str, float],
        top_k: int = 5
    ) -> Tuple[str, str]:
        """
        Generate explanation for an ADMET prediction.
        
        Args:
            smiles: SMILES string of the molecule.
            property_name: ADMET property being predicted.
            prediction: Model prediction value.
            molecular_descriptors: Computed molecular features.
            top_k: Number of context chunks to retrieve.
            
        Returns:
            Tuple of (explanation, retrieved_context).
        """
        context = self.vector_store.get_property_context(
            property_name=property_name,
            molecular_descriptors=molecular_descriptors,
            top_k=top_k
        )
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            smiles=smiles,
            property_name=property_name,
            prediction=prediction,
            molecular_descriptors=molecular_descriptors,
            context=context
        )
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        explanation = response.choices[0].message.content
        
        return explanation, context
    
    def batch_explain(
        self,
        predictions: list,
        property_name: str
    ) -> list:
        """
        Generate explanations for multiple predictions.
        
        Args:
            predictions: List of (smiles, prediction, descriptors) tuples.
            property_name: ADMET property being predicted.
            
        Returns:
            List of explanation strings.
        """
        explanations = []
        
        for smiles, pred, descriptors in predictions:
            explanation, _ = self.explain(
                smiles=smiles,
                property_name=property_name,
                prediction=pred,
                molecular_descriptors=descriptors
            )
            explanations.append(explanation)
        
        return explanations


def get_admet_prediction(
    smiles: str,
    property_name: str,
    model,
    featurizer,
    explainer: Optional[ADMETExplainer] = None
) -> Dict:
    """
    Tool function for LLM to get ADMET prediction.
    
    This function is designed to be called by an LLM as a tool
    to retrieve predictions from the deep learning model.
    
    Args:
        smiles: SMILES string of the molecule to predict.
        property_name: ADMET property to predict (e.g., 'Caco2_Wang').
        model: Trained GNN model.
        featurizer: MolecularFeaturizer instance.
        explainer: Optional ADMETExplainer for interpretations.
        
    Returns:
        Dictionary containing:
            - prediction: float prediction value
            - property: property name
            - smiles: input SMILES
            - descriptors: molecular descriptors
            - explanation: optional explanation text
    """
    graph = featurizer.smiles_to_graph(smiles)
    if graph is None:
        return {"error": f"Failed to parse SMILES: {smiles}"}
    
    descriptors = featurizer.get_molecular_descriptors(smiles)
    
    import torch
    from torch_geometric.data import Batch
    
    batch = Batch.from_data_list([graph])
    
    model.eval()
    with torch.no_grad():
        prediction = model(batch).item()
    
    result = {
        "prediction": prediction,
        "property": property_name,
        "smiles": smiles,
        "descriptors": descriptors,
    }
    
    if explainer is not None:
        explanation, context = explainer.explain(
            smiles=smiles,
            property_name=property_name,
            prediction=prediction,
            molecular_descriptors=descriptors
        )
        result["explanation"] = explanation
        result["context_sources"] = context
    
    return result
