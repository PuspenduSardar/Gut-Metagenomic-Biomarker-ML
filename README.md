# 🧬 Gut Metagenomic Biomarker & ML Explorer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gut-metagenomic-biomarker-ml-bczjrfqo9nzqoyc766uwmu.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

An end-to-end machine learning pipeline and interactive web application for predicting patient disease state (Melanoma cohort) and identifying functional KEGG pathway biomarkers from gut microbiome profiles.

---

**Key Features**
* **Dimensionality Reduction:** Aggregates sparse KEGG Orthologs (KOs) into higher-order functional pathways using the KEGG REST API to mitigate high-dimensionality ($p \gg n$) noise.
* **Compositional Normalization:** Applies Centered Log-Ratio (CLR) transformations to eliminate relative abundance constraints.
* **Leak-Free Cross-Validation:** Runs Recursive Feature Elimination (RFE) strictly *inside* training folds to prevent feature selection bias and data leakage.
* **Independent External Validation:** Tests a locked XGBoost classifier on an unseen validation cohort ($N=56$) to prove true out-of-sample model robustness.
* **Explainable AI (XAI):** Integrates SHAP (SHapley Additive exPlanations) values to visualize feature importance and pathway directionality (Case vs. Control enrichment).

---

**Repository Structure**
```text
.
├── app.py                 # Streamlit interactive dashboard application
├── collapse_kegg.py       # Script for aggregating KOs into KEGG Pathways
├── pipeline_SHAP.py       # Leak-free CV, XGBoost training, and SHAP pipeline
├── requirements.txt       # Cloud deployment dependencies for Streamlit
├── environment.yml        # Conda environment definition for local setup
├── results_dir/           # Pre-computed evaluation metrics and visual plots
└── README.md              # Project documentation


Quickstart Guide

1. Clone Repository & Set Up Environment

git clone [https://github.com/PuspenduSardar/Gut-Metagenomic-Biomarker-ML.git](https://github.com/PuspenduSardar/Gut-Metagenomic-Biomarker-ML.git)
cd Gut-Metagenomic-Biomarker-ML
conda env create -f environment.yml
conda activate metagenomics_ml

2. Collapse KOs & Execute Pipeline

# Collapse KEGG IDs to Pathways
python collapse_kegg.py --input discovery_KO.tsv --output discovery_pathways.tsv
python collapse_kegg.py --input validation_KO.tsv --output validation_pathways.tsv

# Run Leak-Free CV & Independent Validation
python pipeline_SHAP.py \
  --disc_data discovery_pathways.tsv \
  --disc_meta discovery_metadata.txt \
  --val_data validation_pathways.tsv \
  --val_meta validation_metadata.txt \
  --output results_dir/ \
  --n_features 15


3. Launch Streamlit Web App

streamlit run app.py
