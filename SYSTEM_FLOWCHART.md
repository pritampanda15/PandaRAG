# ADMET Prediction System - Architecture Flowcharts

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        A[TDC Dataset] --> B[Data Preprocessing]
        B --> C[Train/Val/Test Split]
    end

    subgraph "Feature Engineering"
        C --> D[Molecular Featurizer]
        D --> E[Graph Construction]
        E --> F[Node Features + Edge Features]
    end

    subgraph "Model Training"
        F --> G[GNN Model<br/>GCN/GAT/GraphSAGE]
        G --> H[Training Loop]
        H --> I[Validation]
        I --> J{Early Stopping?}
        J -->|Continue| H
        J -->|Stop| K[Save Best Model]
    end

    subgraph "Knowledge Base"
        L[Scientific Documents] --> M[Text Chunking]
        M --> N[Embedding Generation]
        N --> O[Vector Store<br/>ChromaDB]
    end

    subgraph "Inference & Explanation"
        P[Input SMILES] --> Q[Featurization]
        Q --> R[GNN Prediction]
        K --> R
        R --> S[Numerical Result]
        S --> T{Explanation<br/>Requested?}
        T -->|No| U[Return Prediction]
        T -->|Yes| V[RAG Retrieval]
        O --> V
        V --> W[LLM Generation<br/>GPT-4o-mini]
        W --> X[Explanation + Context]
    end

    subgraph "User Interfaces"
        Y[Streamlit Web App]
        Z[CLI Script]
    end

    U --> Y
    U --> Z
    X --> Y
    X --> Z

    style G fill:#ff9999
    style O fill:#99ccff
    style W fill:#99ff99
    style K fill:#ffcc99
```

## 2. Training Pipeline Flow

```mermaid
flowchart TD
    Start([Start Training]) --> A[Load Configuration<br/>config.yaml]
    A --> B[Download/Load Dataset<br/>from TDC]
    B --> C[Display Dataset Statistics]

    C --> D[Split Data<br/>70% Train, 10% Val, 20% Test]
    D --> E[Initialize Featurizer]

    E --> F[Featurize Training Set]
    F --> G[Featurize Validation Set]
    G --> H[Featurize Test Set]

    H --> I[Create DataLoaders<br/>Batch Size: 32]

    I --> J[Initialize GNN Model]
    J --> K[Setup Optimizer<br/>Adam + Weight Decay]
    K --> L[Setup LR Scheduler<br/>ReduceLROnPlateau]

    L --> M[Initialize MLflow Tracking]

    M --> N{Epoch < Max Epochs?}
    N -->|Yes| O[Training Epoch]

    O --> P[Forward Pass<br/>Compute Loss MSE/BCE]
    P --> Q[Backward Pass<br/>Gradient Clipping]
    Q --> R[Update Weights]

    R --> S[Validation Epoch]
    S --> T[Compute Metrics<br/>RMSE, MAE, R2, AUROC]

    T --> U[Log to MLflow]
    U --> V[Update LR Scheduler]

    V --> W{Val Loss<br/>Improved?}
    W -->|Yes| X[Save Checkpoint<br/>best_model.pt]
    W -->|No| Y[Increment Patience]

    X --> Z{Patience<br/>Exceeded?}
    Y --> Z

    Z -->|No| N
    Z -->|Yes| AA[Early Stopping Triggered]

    N -->|No| AA
    AA --> AB[Evaluate on Test Set]
    AB --> AC[Log Final Metrics]
    AC --> End([Training Complete])

    style J fill:#ff9999
    style X fill:#99ff99
    style AA fill:#ffcc99
```

## 3. Prediction Pipeline Flow

```mermaid
flowchart TD
    Start([User Input]) --> A{Input Type?}

    A -->|Single SMILES| B[Parse SMILES String]
    A -->|CSV File| C[Load CSV File<br/>Extract SMILES Column]

    B --> D[Validate SMILES]
    C --> E[Validate All SMILES]

    D --> F{Valid?}
    E --> F

    F -->|No| G[Return Error Message]
    F -->|Yes| H[Load Featurizer]

    H --> I[Load Trained Model<br/>best_model.pt]
    I --> J[Set Model to Eval Mode]

    J --> K[Convert SMILES to Graph]
    K --> L[Extract Node Features<br/>Atom Type, Charge, etc.]
    L --> M[Extract Edge Features<br/>Bond Type, Aromaticity]

    M --> N[Create PyG Data Object]
    N --> O[Forward Pass Through GNN]

    O --> P{Task Type?}
    P -->|Regression| Q[Raw Output Value]
    P -->|Classification| R[Apply Sigmoid<br/>Convert to Probability]

    Q --> S[Compute Descriptors<br/>MW, LogP, TPSA]
    R --> S

    S --> T{Explanation<br/>Requested?}

    T -->|No| U[Format Results]
    T -->|Yes| V[Initialize RAG System]

    V --> W[Build Query from<br/>Prediction + Descriptors]
    W --> X[Retrieve Relevant Context<br/>from Vector Store]

    X --> Y[Build LLM Prompt<br/>System + User + Context]
    Y --> Z[Call OpenAI API<br/>GPT-4o-mini]

    Z --> AA[Generate Explanation]
    AA --> AB[Combine Prediction + Explanation]

    U --> AC[Display/Save Results]
    AB --> AC

    AC --> End([Complete])

    G --> End

    style I fill:#ff9999
    style X fill:#99ccff
    style Z fill:#99ff99
```

## 4. RAG System Flow (Knowledge Retrieval & Explanation)

```mermaid
flowchart TD
    subgraph "Knowledge Base Building"
        A[Scientific Documents<br/>PDF, TXT, MD] --> B[Document Loader]
        B --> C[Text Splitter<br/>Chunk Size: 1000<br/>Overlap: 200]
        C --> D[Generate Embeddings<br/>MiniLM-L6-v2]
        D --> E[Store in ChromaDB<br/>with Metadata]
    end

    subgraph "Retrieval Process"
        F[Prediction Context<br/>SMILES + Value + Descriptors] --> G[Build Query String]
        G --> H[Embed Query<br/>Same Model: MiniLM-L6-v2]
        E --> I[Vector Search<br/>Cosine Similarity]
        H --> I
        I --> J[Retrieve Top-K Chunks<br/>K = 3]
        J --> K[Rerank by Relevance]
    end

    subgraph "LLM Explanation Generation"
        K --> L[System Prompt<br/>Pharmacokinetics Expert]
        F --> M[User Prompt<br/>Molecule + Prediction Details]
        L --> N[Combine Prompts + Context]
        M --> N
        N --> O[OpenAI API Call<br/>Temperature: 0.3<br/>Max Tokens: 500]
        O --> P[Generated Explanation]
        P --> Q[Extract Key Points]
    end

    Q --> R[Return to User<br/>Explanation + Sources]

    style E fill:#99ccff
    style O fill:#99ff99
    style R fill:#ffcc99
```

## 5. Streamlit Web Application Flow

```mermaid
flowchart TD
    Start([User Opens App]) --> A[Load Resources<br/>Cache with st.cache_resource]

    A --> B[Load Configuration]
    B --> C[Initialize Featurizer]
    C --> D[Load GNN Model]
    D --> E[Initialize Vector Store]

    E --> F{OPENAI_API_KEY<br/>Available?}
    F -->|Yes| G[Initialize Explainer]
    F -->|No| H[Explainer = None]

    G --> I[Display UI]
    H --> I

    I --> J[Show Sidebar Options]
    J --> K[Property Selection Dropdown]
    K --> L[Input Method Selection<br/>Single SMILES or Batch CSV]

    L --> M{Input Method?}

    M -->|Single| N[Text Input for SMILES]
    M -->|Batch| O[File Upload Widget]

    N --> P[Predict Button Clicked?]
    O --> Q[Process CSV File]

    P -->|No| P
    P -->|Yes| R[Featurize Molecule]

    Q --> S[Batch Predictions Loop]
    S --> R

    R --> T[Model Forward Pass]
    T --> U[Compute Descriptors]

    U --> V{Explanation<br/>Enabled?}
    V -->|Yes| W[RAG Explanation]
    V -->|No| X[Skip Explanation]

    W --> Y[Display Results]
    X --> Y

    Y --> Z[Show Metrics Cards<br/>Prediction, Classification, Unit]
    Z --> AA[Show Descriptors<br/>MW, LogP, TPSA, etc.]

    AA --> AB{Has Explanation?}
    AB -->|Yes| AC[Display Explanation Text<br/>+ Retrieved Context]
    AB -->|No| AD[Show Basic Results Only]

    AC --> AE[Visualization Section<br/>Molecule Structure]
    AD --> AE

    AE --> AF{Batch Mode?}
    AF -->|Yes| AG[Download CSV Button]
    AF -->|No| AH[Wait for Next Input]

    AG --> AH
    AH --> P

    style D fill:#ff9999
    style G fill:#99ff99
    style Y fill:#ffcc99
```

## 6. Molecular Featurization Process

```mermaid
flowchart LR
    A[SMILES String] --> B[RDKit Mol Object]

    B --> C{Valid<br/>Molecule?}
    C -->|No| D[Return Error]
    C -->|Yes| E[Extract Atoms]

    E --> F[Node Feature Matrix]
    F --> F1[Atom Type<br/>One-Hot Encoded]
    F --> F2[Degree<br/>Number of Neighbors]
    F --> F3[Formal Charge]
    F --> F4[Hybridization<br/>sp, sp2, sp3]
    F --> F5[Aromaticity<br/>Boolean]
    F --> F6[Hydrogen Count]

    E --> G[Extract Bonds]
    G --> H[Edge Index<br/>COO Format]
    G --> I[Edge Features]

    I --> I1[Bond Type<br/>Single, Double, Triple]
    I --> I2[Is Conjugated]
    I --> I3[Is in Ring]
    I --> I4[Stereo Config]

    F1 --> J[Combine Node Features]
    F2 --> J
    F3 --> J
    F4 --> J
    F5 --> J
    F6 --> J

    I1 --> K[Combine Edge Features]
    I2 --> K
    I3 --> K
    I4 --> K

    J --> L[PyG Data Object]
    H --> L
    K --> L

    L --> M[Graph Representation<br/>Ready for GNN]

    style B fill:#99ccff
    style L fill:#99ff99
    style M fill:#ffcc99
```

## 7. GNN Model Architecture Flow

```mermaid
flowchart TD
    A[Input Graph<br/>Data Object] --> B[Node Features<br/>Tensor: N x F_in]
    A --> C[Edge Index<br/>Tensor: 2 x E]
    A --> D[Edge Features<br/>Tensor: E x F_edge]

    B --> E[GNN Layer 1<br/>GCN/GAT/GraphSAGE]
    C --> E
    D --> E

    E --> F[Batch Normalization]
    F --> G[ReLU Activation]
    G --> H[Dropout 0.2]

    H --> I[GNN Layer 2]
    C --> I

    I --> J[Batch Normalization]
    J --> K[ReLU Activation]
    K --> L[Dropout 0.2]

    L --> M[GNN Layer 3]
    C --> M

    M --> N[Batch Normalization]
    N --> O[ReLU Activation]
    O --> P[Dropout 0.2]

    P --> Q{Pooling Strategy}

    Q -->|Mean| R[Global Mean Pool]
    Q -->|Max| S[Global Max Pool]
    Q -->|Add| T[Global Add Pool]

    R --> U[Graph-Level Embedding<br/>Tensor: B x Hidden_dim]
    S --> U
    T --> U

    U --> V[Fully Connected Layer 1<br/>Hidden → Hidden/2]
    V --> W[ReLU + Dropout]
    W --> X[Fully Connected Layer 2<br/>Hidden/2 → Output_dim]

    X --> Y{Task Type?}
    Y -->|Regression| Z[Raw Output<br/>Continuous Value]
    Y -->|Classification| AA[Logit Output<br/>Apply Sigmoid Later]

    Z --> AB[Final Prediction]
    AA --> AB

    style E fill:#ff9999
    style I fill:#ff9999
    style M fill:#ff9999
    style U fill:#99ff99
    style AB fill:#ffcc99
```

## Component Descriptions

### Key Components:

1. **Data Layer**: Handles dataset loading from TDC and preprocessing
2. **Featurizer**: Converts SMILES to graph representations with molecular features
3. **GNN Model**: Graph neural network for molecular property prediction
4. **Training Loop**: Manages epochs, validation, early stopping, and checkpointing
5. **Vector Store**: ChromaDB storage for knowledge base embeddings
6. **RAG System**: Retrieves relevant context and generates explanations
7. **Interfaces**: Streamlit web app and CLI scripts for user interaction

### File Mappings:

- `scripts/train_model.py` → Training Pipeline
- `scripts/predict.py` → CLI Prediction Pipeline
- `app/main.py` → Streamlit Application Flow
- `src/data/featurizer.py` → Molecular Featurization
- `src/models/gnn.py` → GNN Architecture
- `src/models/trainer.py` → Training Loop
- `src/rag/vector_store.py` → Knowledge Base Management
- `src/rag/llm_explainer.py` → RAG Explanation System

---

*Generated for ADMET Prediction System - Graph Neural Networks with RAG*
