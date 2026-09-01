import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Metagenomic Biomarker Explorer", page_icon="🧬", layout="wide"
)

st.title("🧬 Gut Metagenomic Biomarker & ML Explorer")
st.markdown(
    "Predicting disease state and identifying functional KEGG pathway biomarkers from gut microbiome profiles."
)

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["Overview & Performance", "Biomarker Discovery", "Interactive Predictor"],
)

OUTPUT_DIR = "results_dir"

if page == "Overview & Performance":
    st.header("1. Model Performance & Validation")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("5-Fold Cross-Validation (Discovery)")
        cv_file = os.path.join(OUTPUT_DIR, "cv_metrics.txt")
        if os.path.exists(cv_file):
            with open(cv_file) as f:
                st.code(f.read())
        roc_img = os.path.join(OUTPUT_DIR, "cv_roc_curve.png")
        if os.path.exists(roc_img):
            st.image(
                roc_img, caption="Leak-Free 5-Fold Cross-Validation ROC Curve"
            )

    with col2:
        st.subheader("Independent Validation Cohort")
        val_file = os.path.join(OUTPUT_DIR, "validation_metrics.txt")
        if os.path.exists(val_file):
            with open(val_file) as f:
                st.code(f.read())

elif page == "Biomarker Discovery":
    st.header("2. Top Functional Biomarkers & Directionality")

    tsv_path = os.path.join(OUTPUT_DIR, "feature_directionality.tsv")
    if os.path.exists(tsv_path):
        df_feat = pd.read_csv(tsv_path, sep="\t")
        st.dataframe(df_feat, use_container_width=True)

        st.subheader("SHAP Feature Importance & Directionality")
        shap_img = os.path.join(OUTPUT_DIR, "shap_summary.png")
        if os.path.exists(shap_img):
            st.image(
                shap_img,
                caption="SHAP Impact on Model Predictions (Case vs. Control)",
            )

elif page == "Interactive Predictor":
    st.header("3. Single-Sample Prediction Simulator")
    st.markdown(
        "Run real-time disease prediction on new, CLR-transformed pathway profiles using the trained XGBoost model."
    )

    model_path = os.path.join(OUTPUT_DIR, "xgboost_model.json")
    feat_path = os.path.join(OUTPUT_DIR, "selected_features.json")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    demo_path = os.path.join(OUTPUT_DIR, "demo_sample.json")

    if os.path.exists(model_path) and os.path.exists(feat_path):
        clf = XGBClassifier()
        clf.load_model(model_path)

        with open(feat_path) as f:
            selected_features = json.load(f)

        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        st.subheader("Choose Input Method")
        input_mode = st.radio(
            "Select how to feed sample data:",
            ["Use Demo Sample", "Upload Custom CSV/TSV"],
        )

        sample_df = None

        if input_mode == "Use Demo Sample" and os.path.exists(demo_path):
            sample_df = pd.read_json(demo_path)
            # Ensure column order matches selected features
            sample_df = sample_df.reindex(columns=selected_features)
            st.success("Loaded pre-formatted CLR demo sample from validation cohort.")

        elif input_mode == "Upload Custom CSV/TSV":
            uploaded_file = st.file_uploader(
                "Upload CLR-transformed TSV/CSV file",
                type=["tsv", "csv", "txt"],
            )
            if uploaded_file:
                sep = "\t" if uploaded_file.name.endswith((".tsv", ".txt")) else ","
                user_df = pd.read_csv(uploaded_file, sep=sep, index_col=0)

                # Clean header whitespace
                user_df.columns = user_df.columns.astype(str).str.strip()
                user_df.index = user_df.index.astype(str).str.strip()

                # Detect matrix orientation (Pathways as rows vs columns)
                if not any(feat in user_df.columns for feat in selected_features):
                    if any(feat in user_df.index for feat in selected_features):
                        user_df = user_df.T

                # Reindex to extract 15 target features
                sample_df = user_df.reindex(
                    columns=selected_features, fill_value=0
                ).iloc[0:1]

                # Check for feature matching
                missing_feats = [
                    f for f in selected_features if f not in user_df.columns
                ]
                if len(missing_feats) == len(selected_features):
                    st.error(
                        "⚠️ None of the required 15 pathway signatures were found in the uploaded file. Please verify feature names."
                    )
                elif missing_feats:
                    st.warning(
                        f"⚠️ {len(missing_feats)} missing pathway features were defaulted to 0.0."
                    )

        if sample_df is not None:
            st.subheader("Input CLR Feature Abundances (Unscaled)")
            st.dataframe(sample_df, use_container_width=True)

            if st.button("🚀 Run Prediction"):
                # Apply Discovery Cohort Z-score Scaling before prediction
                if scaler is not None:
                    sample_scaled = pd.DataFrame(
                        scaler.transform(sample_df),
                        columns=selected_features,
                        index=sample_df.index,
                    )
                else:
                    sample_scaled = sample_df
                    st.warning(
                        "Scaler file not found. Running prediction on unscaled data."
                    )

                prob = clf.predict_proba(sample_scaled)[0][1]
                pred_class = clf.predict(sample_scaled)[0]

                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)

                with col1:
                    if pred_class == 1:
                        st.error("### Result: **Responder / Case (Class 1)**")
                        st.caption(
                            "High probability match to target disease/responder phenotype."
                        )
                    else:
                        st.success("### Result: **Non-Responder / Control (Class 0)**")
                        st.caption(
                            "High probability match to baseline control phenotype."
                        )

                with col2:
                    st.metric(
                        label="Disease Probability Score", value=f"{prob:.1%}"
                    )
                    st.progress(float(prob))

                # Diagnostic Inspector for Validation
                with st.expander("🔍 Model Diagnostic & Feature Inspector"):
                    st.markdown("**1. Scaled Z-Score Input to XGBoost Model**")
                    st.dataframe(sample_scaled, use_container_width=True)

                    st.markdown("**2. Summary Metrics**")
                    st.write(f"- **Raw CLR Mean:** {sample_df.values.mean():.4f}")
                    st.write(
                        f"- **Z-Score Transformed Mean:** {sample_scaled.values.mean():.4f}"
                    )
                    st.write(
                        f"- **Raw Model Output (Class 1 Probability):** {prob:.4f}"
                    )
    else:
        st.warning(
            "Model files not found. Please ensure results_dir/ contains xgboost_model.json and scaler.joblib."
        )