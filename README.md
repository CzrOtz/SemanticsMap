# GraphIt: High-Dimensional Semantic Manifold Laboratory

## Executive Summary

GraphIt is an analytical application designed to project high-dimensional text embeddings into interpretable 3D to 6D coordinate spaces. Built for data scientists and researchers, this tool facilitates semantic cluster analysis, document overlap detection, and latent thematic structure visualization using state-of-the-art manifold learning algorithms.

By mapping complex vector relationships (such as 384-dimensional Sentence Transformer outputs) into a visible coordinate system, GraphIt allows users to mathematically audit semantic similarities, analyze centroid drift, and interrogate the underlying geometry of language data.

---

## Core Analytical Capabilities

### 1. Dynamic Dimensionality Reduction

The application provides fully tunable implementations of industry-standard reduction algorithms to manage local versus global data structure preservation:

- **PaCMAP** (Pairwise Controlled Manifold Approximation) for superior global structure retention.
- **UMAP** (Uniform Manifold Approximation and Projection) for topological data analysis.
- **PCA** (Principal Component Analysis) for linear transformations.
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding) limited to 3D via the Barnes-Hut method for optimized local clustering.

### 2. Multi-Dimensional Visual Encoding

GraphIt utilizes Plotly to render complex relationships beyond standard 3D scatter plots. The visualization engine maps high-dimensional data using the following encoding logic:

| Dimensions | Encoding Method |
|------------|-----------------|
| **3D** | Spatial mapping utilizing X, Y, and Z coordinates. |
| **4D** | Spatial mapping + continuous color scales (dim4). |
| **5D** | Spatial mapping + color scale + absolute magnitude represented by marker size (dim5). |
| **6D** | Vector Cone Plots utilizing spatial position (X, Y, Z) and trajectory vectors (U, V, W). |

### 3. Pre-Reduction Analytics

Before manifold reduction is applied, the laboratory processes the raw high-dimensional embeddings to provide:

- **Cosine Similarity Matrices:** Exhaustive pairwise distance calculations between document clusters.
- **Grand Tour Projections:** Dynamic, continuous rotation of the high-dimensional data to reveal structural patterns prior to dimensional collapse.

### 4. Automated Text Sanitization Pipeline

High-quality vectorization relies on pristine input data. The `utils.py` ingestion engine features configurable sanitization utilizing `cleantext` and Regex:

- Unicode to ASCII translation.
- Automated removal of URLs, emails, and malformed metadata.
- Bracket, parenthesis, and special character stripping.

---

## Technical Architecture & Stack

| Component | Technology |
|-----------|------------|
| **Application Framework** | Streamlit |
| **Language Models** | sentence-transformers (Hugging Face ecosystem including MiniLM, MPNet, BGE, and E5 architectures) |
| **Mathematical Operations** | numpy, pandas, scipy |
| **Manifold Learning** | scikit-learn, umap-learn, pacmap |
| **Visualization Engine** | plotly |
| **Tokenization** | nltk |

---

## Repository Structure
```plaintext
GraphIt/
├── .gitignore                   # Version control exclusion rules
├── README.md                    # Technical documentation
├── requirements.txt             # Dependency constraints and environment replication
├── semantic_similarity_lab.py   # Main Streamlit application and UI routing
├── process_functions.py         # Visualization engine and metric calculation logic
└── utils.py                     # Data ingestion, text sanitization, and reduction math
```

---

## Installation & Environment Setup

> This project requires **Python 3.9 or higher**. It is highly recommended to use a virtual environment to ensure reproducibility.

### 1. Clone the repository
```bash
git clone https://github.com/CzrOtz/GraphIt.git
cd GraphIt
```

### 2. Initialize virtual environment *(Optional but recommended)*
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the application
```bash
python -m streamlit run semantic_similarity_lab.py
```

---

## Operational Workflow

1. **Model Selection:** Select the desired Hugging Face embedding model from the sidebar. Multilingual support is automatically detected and indicated based on the chosen model architecture.

2. **Algorithm Tuning:** Select a reduction algorithm (PaCMAP, UMAP, PCA, t-SNE) and configure the hyperparameters (e.g., `n_neighbors`, learning rate, minimum distance) via the sidebar expanders.

3. **Data Ingestion:** Paste source text into the provided document containers. A minimum of two sources is required for cross-document cosine similarity matrices, and a minimum of three sentences per source is required for manifold reduction.

4. **Execution:** Select the processing scope (Pre-reduction, Post-reduction, or Both) and execute the pipeline. The application will generate interactive dataframes, metric summaries (centroids, standard deviations, magnitude extremes), and Plotly figures.
