# 🧬 Gut Metagenomic Biomarker & ML Explorer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gut-metagenomic-biomarker-ml-bczjrfqo9nzqoyc766uwmu.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

An end-to-end machine learning pipeline and interactive web application for predicting patient disease state (Melanoma cohort) and identifying functional KEGG pathway biomarkers from gut microbiome profiles.

---


---

## 📌 Overview

This repository hosts the interactive deployment interface for our metagenomic biomarker discovery pipeline. The application allows researchers and clinical data scientists to:
1. **Explore Model Performance:** Review cross-validation metrics and external validation performance ($N=56$ unseen validation cohort) obtained using a leak-free XGBoost workflow.
2. **Examine Biomarkers:** Analyze top functional KEGG pathways prioritized using SHAP (SHapley Additive exPlanations) feature directionality.
3. **Simulate Clinical Predictions:** Run real-time single-sample inference using custom patient data or a pre-loaded validation demo sample.

---

## 📁 Repository Structure

```text
.
├── app.py                 # Interactive dashboard application
├── collapse_kegg.py       # Script for aggregating raw KOs into KEGG Pathways
├── requirements.txt       # Python dependencies for deployment
├── packages.txt           # C++ system dependencies for Linux containers (libgomp1)
├── results_dir/           # Trained XGBoost model, selected features, & metric visual outputs
├── LICENSE                # MIT License
└── README.md              # Project documentation


🧪 Data Preparation Workflow for External Inputs
To test custom patient samples on the Interactive Predictor tab, input data must undergo specific preprocessing steps to match the model's training distribution.

[Raw KO Counts] ──> 1. Collapse to Pathways ──> 2. Full-Profile CLR Transform ──> 3. Upload CSV/TSV


Step 1: Aggregate KOs to KEGG Pathways
Convert raw KEGG Ortholog (KO) abundance matrices into pathway-level abundances using the provided script:

python collapse_kegg.py --input your_sample_KO.tsv --output your_sample_pathways.tsv

Step 2: Apply Centered Log-Ratio (CLR) TransformationTo eliminate relative abundance constraints ($p \gg n$ compositionality), apply a Centered Log-Ratio (CLR) transformation across the full pathway profile:

```math
$\text{CLR}(x_i) = \ln\left(\frac{x_i}{g(\mathbf{x})}\right)$
```

⚠️ Critical Requirement: CLR must be calculated on the entire background pathway table before filtering down to selected features. Calculating CLR on a subset of features distorts the sample's geometric mean $g(\mathbf{x})$ and produces invalid model inputs.

Step 3: Single-Sample Upload
Upload the CLR-transformed file (CSV/TSV) containing pathway features (rows or columns). The app automatically extracts the required 15 pathway signatures, sets missing background features to zero, and outputs a disease probability score.

🚀 Running the Web App Locally
If you prefer running the dashboard on your local machine:

1. Clone the repository:
git clone [https://github.com/PuspenduSardar/Gut-Metagenomic-Biomarker-ML.git](https://github.com/PuspenduSardar/Gut-Metagenomic-Biomarker-ML.git)
cd Gut-Metagenomic-Biomarker-ML

2. Install dependencies:
pip install -r requirements.txt

3. Launch Streamlit:
streamlit run app.py

📄 License
Distributed under the MIT License. See LICENSE for more information.
