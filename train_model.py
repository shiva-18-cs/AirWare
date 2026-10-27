"""
AirAware: Pollution-Risk Classification & Model Training
--------------------------------------------------------
B.Tech Computer Engineering Machine Learning Mini Project.
Trains a Random Forest Classifier to predict CPCB pollution-risk
categories (AQI_Bucket) from chemical pollutant concentrations.

Leak Prevention Rules:
1. Target column (AQI_Bucket) and continuous AQI are excluded from features.
2. Missing values are imputed using median values computed ONLY on X_train.
3. Model is evaluated on an unseen 20% stratified test partition.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Define paths and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "city_day.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "pollution_classifier.joblib")

# Selected chemical pollutant features available in the dataset
FEATURE_COLS = ["PM2.5", "PM10", "NO2", "NH3", "CO", "SO2", "O3"]
TARGET_COL = "AQI_Bucket"

# Official CPCB categories in logical severity order
CPCB_CLASSES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def load_and_preprocess_data():
    """Loads city_day.csv, validates columns, drops rows with missing target."""
    print("=" * 65)
    print("AirAware: Data Loading & Preprocessing")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Primary dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded raw dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Verify required feature columns exist
    missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Remove rows where target label is missing
    initial_count = len(df)
    df_clean = df.dropna(subset=[TARGET_COL]).copy()
    dropped_count = initial_count - len(df_clean)
    print(f"Dropped {dropped_count:,} rows with missing '{TARGET_COL}'.")
    print(f"Valid labeled samples available: {len(df_clean):,}")

    # Inspect class distribution
    print("\nTarget Class Distribution:")
    for cls_name, count in df_clean[TARGET_COL].value_counts().items():
        print(f" - {cls_name:<15}: {count:,} samples ({count / len(df_clean) * 100:.2f}%)")

    return df_clean


def train_and_evaluate():
    """Fits imputer and Random Forest on training data, evaluates on test data."""
    df = load_and_preprocess_data()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # 80:20 Stratified Split to preserve class proportions
    print("\n" + "=" * 65)
    print("Train/Test Stratified Split (80% Train, 20% Test, random_state=42)")
    print("=" * 65)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples : {len(X_test):,}")

    # Median Imputer fitted strictly on training partition
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Compute training medians for default UI inputs
    train_medians = dict(zip(FEATURE_COLS, [round(float(v), 2) for v in imputer.statistics_]))

    # Train Random Forest Classifier
    print("\nTraining Random Forest Classifier (100 estimators, random_state=42)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_imp, y_train)
    print("Model training complete.")

    # Evaluate on held-out test data
    y_pred = clf.predict(X_test_imp)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Reorder classes consistently to match CPCB logical severity order
    existing_classes = [c for c in CPCB_CLASSES if c in clf.classes_]
    cm = confusion_matrix(y_test, y_pred, labels=existing_classes)

    # Extract feature importances
    feature_importances = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in clf.feature_importances_]))
    # Sort feature importances descending
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    print("\n" + "=" * 65)
    print("Evaluation Results on Held-Out Test Data:")
    print("=" * 65)
    print(f"  Test Accuracy   : {accuracy * 100:.2f}%")
    print(f"  Macro Precision : {precision * 100:.2f}%")
    print(f"  Macro Recall    : {recall * 100:.2f}%")
    print(f"  Macro F1-Score  : {f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("\nFeature Importances:")
    for feat, imp in sorted_importances.items():
        print(f"  - {feat:<8}: {imp * 100:.2f}%")

    # Package all necessary components into a single clean dictionary artifact
    artifact = {
        "model": clf,
        "imputer": imputer,
        "feature_names": FEATURE_COLS,
        "target_name": TARGET_COL,
        "classes": existing_classes,
        "train_medians": train_medians,
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "precision_macro": round(float(precision), 4),
            "recall_macro": round(float(recall), 4),
            "f1_macro": round(float(f1), 4),
            "test_sample_count": len(y_test),
            "train_sample_count": len(y_train),
        },
        "confusion_matrix": cm.tolist(),
        "feature_importances": sorted_importances
    }

    joblib.dump(artifact, MODEL_SAVE_PATH, compress=3)
    print(f"\n[OK] Model artifact successfully saved to:\n  -> {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
