# ADMET Property Prediction System

A deep learning system that predicts Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET) properties from molecular structures. The system combines Graph Neural Networks with a RAG-based LLM explainer to provide both predictions and mechanistic interpretations.

## Project Overview

| Component | Description |
|-----------|-------------|
| Domain | Preclinical Drug Development |
| Core Task | Predict ADMET properties from SMILES molecular representations |
| Model Architecture | Graph Neural Network (GNN) with message passing |
| Explainability | RAG pipeline with pharmacokinetic knowledge base |
| Output | Property predictions with evidence-based explanations |

## ADMET Properties Predicted

| Property | Category | Description |
|----------|----------|-------------|
| Caco-2 Permeability | Absorption | Intestinal absorption potential |
| HIA | Absorption | Human Intestinal Absorption |
| BBB Penetration | Distribution | Blood-Brain Barrier crossing |
| Plasma Protein Binding | Distribution | Drug availability in bloodstream |
| CYP450 Inhibition | Metabolism | Drug-drug interaction risk |
| Half-life | Excretion | Duration in body |
| hERG Inhibition | Toxicity | Cardiac toxicity risk |
| AMES Toxicity | Toxicity | Mutagenicity |

## Project Structure

```
admet-prediction-system/
├── config/
│   └── config.yaml              # Model and pipeline configuration
├── data/
│   ├── raw/                     # Original TDC datasets
│   ├── processed/               # Featurized molecular graphs
│   └── knowledge_base/          # RAG documents
├── src/
│   ├── data/
│   │   ├── data_loader.py       # TDC data fetching
│   │   └── molecular_featurizer.py  # SMILES to graph conversion
│   ├── models/
│   │   ├── gnn_model.py         # Graph Neural Network architecture
│   │   └── trainer.py           # Training loop with MLflow
│   ├── rag/
│   │   ├── document_processor.py    # PDF and text ingestion
│   │   ├── vector_store.py      # ChromaDB embeddings
│   │   └── llm_explainer.py     # Explanation generation
│   └── utils/
│       └── helpers.py           # Utility functions
├── scripts/
│   ├── train_model.py           # Model training entry point
│   ├── build_knowledge_base.py  # RAG index builder
│   └── predict.py               # Inference script
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── app/
│   └── main.py                  # Streamlit application
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/admet-prediction-system.git
cd admet-prediction-system

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` to set:

```yaml
data:
  dataset_name: "Caco2_Wang"
  test_split: 0.2

model:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.2
  learning_rate: 0.001
  epochs: 100

rag:
  embedding_model: "all-MiniLM-L6-v2"
  llm_model: "gpt-4o-mini"
  top_k: 5
```

## Usage

### Step 1: Download Data

```bash
python scripts/train_model.py --download-only
```

### Step 2: Build Knowledge Base

```bash
python scripts/build_knowledge_base.py --docs-path data/knowledge_base/
```

### Step 3: Train Model

```bash
python scripts/train_model.py --config config/config.yaml
```

### Step 4: Run Predictions

```bash
python scripts/predict.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --explain
```

### Step 5: Launch Application

```bash
streamlit run app/main.py
```

## Data Source

This project uses the Therapeutics Data Commons (TDC) benchmark datasets:

Huang, K., et al. "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development." NeurIPS 2021.

Available at: https://tdcommons.ai/

## Model Architecture

The GNN follows a message passing paradigm:

1. Atom features are initialized from molecular properties (atomic number, charge, hybridization)
2. Bond features encode connectivity (bond type, aromaticity)
3. Message passing layers aggregate neighbor information
4. Global pooling produces a molecular fingerprint
5. MLP head outputs property predictions

## RAG Knowledge Sources

The explanation system retrieves from:

1. FDA Drug Labels (DailyMed)
2. PubChem Compound Summaries
3. Pharmacokinetic Textbook Excerpts
4. ADMET Review Articles

## MLflow Tracking

Experiments are logged to MLflow:

```bash
mlflow ui --port 5000
```

Tracked metrics include:
- Training and validation loss
- AUROC for classification tasks
- RMSE for regression tasks
- Model hyperparameters
- Feature importance

## Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

## License

MIT License
