# DVCHN

**DVCHN** (Dual-View Contrastive Hypergraph Network) is a model for **drug-miRNA association prediction**.

It uses two views of the data:
- a **bipartite graph** built from known drug-miRNA associations,
- a **semantic hypergraph** built from feature similarity.

The repository includes the model code, benchmark datasets, and several scripts for additional experiments.

## Repository layout

```text
DVCHN/
├── README.md
├── code/
│   ├── main.py
│   ├── model.py
│   ├── utils.py
│   ├── grid_search_full.py
│   ├── run_sparsity.py
│   ├── cross_predict.py
│   ├── transfer_main.py
│   ├── plot_tsne.py
│   ├── inductive_test.py.py
│   ├── neg_ratio_test.py.py
│   └── sparsity_test.py.py
└── data/
    ├── dataset1/
    ├── dataset2/
    └── llm/
```

## Data

This work uses two benchmark datasets:

- **Dataset 1**: 831 drugs, 541 miRNAs, 664 known associations
- **Dataset 2**: 39 drugs, 286 miRNAs, 664 known associations

Each dataset folder contains:
- biological feature files
- LLM-derived feature files
- the drug-miRNA association matrix

In the current implementation, biological features and LLM-derived features are each aligned to **512 dimensions**, then concatenated into **1024-dimensional** node features.

## Before running

### 1. Rename the dataset folder

The code reads data from:

```text
data/dataset1
data/dataset2
```

So if your folder is currently named `dataset/`, rename it to `data/`.

### 2. Keep the project structure simple

Recommended layout:

```text
DVCHN/
├── code/
└── data/
```

### 3. CUDA is expected by default

`main.py` creates the model with `.cuda()`, so the main training script expects a GPU environment unless you modify the code.

## Environment

The project is implemented with:
- Python
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- scikit-learn
- Matplotlib

Example setup:

```bash
conda create -n dvchn python=3.9
conda activate dvchn
pip install numpy pandas scikit-learn matplotlib
```

Then install **PyTorch** and **PyTorch Geometric** with versions that match your CUDA environment.

## Quick start

### Run 5-fold cross-validation on Dataset 1

```bash
cd code
python main.py --dataset 1 
```

### Run 5-fold cross-validation on Dataset 2

```bash
cd code
python main.py --dataset 2 
```

## Main scripts

### Core files
- `main.py`: main training and evaluation script
- `model.py`: DVCHN model definition
- `utils.py`: data loading, preprocessing, metrics, and hypergraph construction

### Extra experiment scripts
- `grid_search_full.py`: hyperparameter search
- `run_sparsity.py`: sparsity robustness experiment
- `sparsity_test.py.py`: reduced-training-edge experiment
- `neg_ratio_test.py.py`: negative sampling ratio experiment
- `inductive_test.py.py`: strictly held-out independent test
- `cross_predict.py`: prediction on unknown candidate pairs
- `transfer_main.py`: consensus report generation from CSV prediction files
- `plot_tsne.py`: t-SNE visualization of learned embeddings


