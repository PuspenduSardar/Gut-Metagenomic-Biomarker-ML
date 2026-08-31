import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, auc, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


def load_and_prep(data_path, meta_path):
    """Loads pathway abundance and metadata, aligns sample IDs, and applies CLR."""
    # Do NOT use .T if input data is already Samples (rows) x Pathways (cols)
    df = pd.read_csv(data_path, sep="\t", index_col=0).astype(float)
    meta = pd.read_csv(meta_path, sep="\t")
    meta.columns = meta.columns.str.lower()

    # Clean string indices to remove hidden spaces
    df.index = df.index.astype(str).str.strip()

    # Align Sample IDs using the first column of metadata
    sample_col = meta.columns[0]
    meta[sample_col] = meta[sample_col].astype(str).str.strip()
    meta = meta.set_index(sample_col)

    common_samples = df.index.intersection(meta.index)
    print(
        f"Loaded {len(df.index)} data samples and {len(meta.index)} metadata samples."
    )
    print(f"Found {len(common_samples)} overlapping sample IDs.")

    if len(common_samples) == 0:
        print("\n--- DEBUG: Sample ID Mismatch ---")
        print(f"Data sample IDs (first 3):     {list(df.index[:3])}")
        print(f"Metadata sample IDs (first 3): {list(meta.index[:3])}")
        raise ValueError(
            "Zero overlapping sample IDs between data and metadata files."
        )

    df = df.loc[common_samples]
    meta = meta.loc[common_samples]

    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(meta["condition"])

    # CLR Transformation (+ pseudo-count)
    df_clr = np.log(df + 1e-6) - np.log(df + 1e-6).mean(axis=1).values[:, None]
    return pd.DataFrame(df_clr, index=df.index, columns=df.columns), y, le


def run_leak_free_cv(X, y, n_features, output_dir):
    """Runs 5-Fold CV where RFE feature selection is fit ONLY on training folds."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 1. Scale inside fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 2. Select features ONLY on training fold
        base_xgb = XGBClassifier(
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
        rfe = RFE(estimator=base_xgb, n_features_to_select=n_features, step=0.1)
        X_train_sel = rfe.fit_transform(X_train_scaled, y_train)
        X_test_sel = rfe.transform(X_test_scaled)

        # 3. Train final fold model
        clf = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
        clf.fit(X_train_sel, y_train)
        y_score = clf.predict_proba(X_test_sel)[:, 1]

        # 4. Metrics & ROC
        fpr, tpr, _ = roc_curve(y_test, y_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.5, label=f"Fold {i+1} (AUC = {fold_auc:.2f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=2)
    ax.plot(
        mean_fpr,
        np.mean(tprs, axis=0),
        color="b",
        lw=2,
        label=f"Mean ROC (AUC = {np.mean(aucs):.2f} ± {np.std(aucs):.2f})",
    )
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Leak-Free 5-Fold CV ROC Curve",
    )
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "cv_roc_curve.png"))
    plt.close()

    print(f"Realistic Cross-Validation AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")


def evaluate_independent_validation(
    X_disc, y_disc, X_val, y_val, n_features, output_dir
):
    """Trains on full Discovery cohort and tests on independent Validation cohort."""
    # Find overlapping KEGG IDs present in both datasets
    common_features = X_disc.columns.intersection(X_val.columns)
    X_disc = X_disc[common_features]
    X_val = X_val[common_features]

    # Preprocessing & RFE on Full Discovery
    scaler = StandardScaler()
    X_disc_scaled = scaler.fit_transform(X_disc)
    X_val_scaled = scaler.transform(X_val)

    base_xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    rfe = RFE(estimator=base_xgb, n_features_to_select=n_features, step=0.1)
    X_disc_sel = rfe.fit_transform(X_disc_scaled, y_disc)
    X_val_sel = rfe.transform(X_val_scaled)

    # Fit final model & Predict on Validation
    clf = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    clf.fit(X_disc_sel, y_disc)

    y_val_pred = clf.predict(X_val_sel)
    y_val_proba = clf.predict_proba(X_val_sel)[:, 1]

    val_auc = roc_auc_score(y_val, y_val_proba)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)

    print("\n--- Independent Validation Results ---")
    print(f"Validation ROC-AUC:  {val_auc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disc_data", required=True)
    parser.add_argument("--disc_meta", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--val_meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_features", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    X_disc, y_disc, le = load_and_prep(args.disc_data, args.disc_meta)
    X_val, y_val, _ = load_and_prep(args.val_data, args.val_meta)

    run_leak_free_cv(X_disc, y_disc, args.n_features, args.output)
    evaluate_independent_validation(
        X_disc, y_disc, X_val, y_val, args.n_features, args.output
    )
