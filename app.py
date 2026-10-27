"""
AirAware: Explainable Air Pollution Risk Analysis
-------------------------------------------------
B.Tech Computer Engineering Machine Learning Mini Project.
Interactive Streamlit Application featuring:
- Section A: Pollution-Risk Category Prediction (Random Forest Classifier)
- Section B: Pollutant Correlations & Historical City Analysis
- Section C: Empirical Test Evaluation Metrics & Confusion Matrix
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Setup & Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AirAware — Pollution-Risk Analysis",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "city_day.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pollution_classifier.joblib")

# CPCB Category Configuration (Color & Advisory)
CPCB_INFO = {
    "Good": {
        "color": "#28a745",
        "advisory": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "caution": "None. Ideal conditions for outdoor activities."
    },
    "Satisfactory": {
        "color": "#8bc34a",
        "advisory": "Air quality is acceptable; may cause minor breathing discomfort to sensitive individuals.",
        "caution": "Unusually sensitive people should consider reducing prolonged outdoor exertion."
    },
    "Moderate": {
        "color": "#ffc107",
        "advisory": "May cause breathing discomfort to individuals with lung diseases, asthma, or heart conditions.",
        "caution": "Children, the elderly, and individuals with respiratory conditions should limit prolonged outdoor exertion."
    },
    "Poor": {
        "color": "#fd7e14",
        "advisory": "May cause breathing discomfort to most people on prolonged outdoor exposure.",
        "caution": "People with heart or lung disease should avoid outdoor activity; general public should reduce exertion."
    },
    "Very Poor": {
        "color": "#dc3545",
        "advisory": "May cause respiratory illness to healthy adults on prolonged exposure; severe impact on sensitive groups.",
        "caution": "Everyone should avoid outdoor physical exertion. Keep windows closed and wear protective masks."
    },
    "Severe": {
        "color": "#721c24",
        "advisory": "Health emergency! Affects healthy people and seriously impacts individuals with existing medical conditions.",
        "caution": "Stay indoors, avoid all outdoor physical activity, and operate indoor air purifiers."
    }
}

# ---------------------------------------------------------
# Caching Data & Model Loading
# ---------------------------------------------------------
@st.cache_data
def load_historical_data():
    """Loads city_day.csv for correlation and historical city analysis."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_trained_model():
    """Loads saved model artifact from models/pollution_classifier.joblib."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

df_history = load_historical_data()
model_artifact = load_trained_model()

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.title("🌿 AirAware")
st.subheader("Explainable Air Pollution Risk Analysis & Forecasting")
st.caption("B.Tech Computer Engineering Machine Learning Mini Project • Traditional ML (Random Forest, Scikit-Learn, Pandas)")
st.markdown("---")

# ---------------------------------------------------------
# Navigation Sidebar
# ---------------------------------------------------------
st.sidebar.title("AirAware Navigation")
section = st.sidebar.radio(
    "Go to section:",
    [
        "A. Pollution-Risk Prediction",
        "B. Pollutant & City Analysis",
        "C. Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Core Objectives:**\n"
    "1. Classify air into 6 CPCB pollution-risk tiers from pollutant levels.\n"
    "2. Analyze pollutant correlations with AQI.\n"
    "3. Identify historical urban pollution hotspots."
)

# ---------------------------------------------------------
# SECTION A: POLLUTION-RISK PREDICTION
# ---------------------------------------------------------
if section == "A. Pollution-Risk Prediction":
    st.header("🧮 Section A: Pollution-Risk Category Prediction")
    st.write(
        "Enter measured chemical pollutant concentrations to predict the official "
        "**CPCB pollution-risk category** using a trained Random Forest Classifier."
    )

    if model_artifact is None:
        st.error(
            "⚠️ Model artifact not found! Please train the model first by running:\n\n"
            "```powershell\npython train_model.py\n```"
        )
    else:
        # Extract metadata from artifact
        train_defaults = model_artifact.get("train_medians", {})
        feature_cols = model_artifact.get("feature_names", ["PM2.5", "PM10", "NO2", "NH3", "CO", "SO2", "O3"])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Particulate Pollutants")
            pm25 = st.number_input(
                "PM2.5 (Fine Particulate Matter, µg/m³)",
                min_value=0.0, max_value=1000.0,
                value=float(train_defaults.get("PM2.5", 55.0)),
                step=1.0,
                help="Fine inhalable particles with diameters 2.5 micrometers and smaller."
            )
            pm10 = st.number_input(
                "PM10 (Coarse Particulate Matter, µg/m³)",
                min_value=0.0, max_value=1200.0,
                value=float(train_defaults.get("PM10", 95.0)),
                step=1.0,
                help="Inhalable particles with diameters 10 micrometers and smaller."
            )
            no2 = st.number_input(
                "NO2 (Nitrogen Dioxide, µg/m³)",
                min_value=0.0, max_value=400.0,
                value=float(train_defaults.get("NO2", 22.0)),
                step=0.5,
                help="Gaseous pollutant primarily emitted from vehicular fuel combustion."
            )

        with col2:
            st.markdown("#### Gaseous Pollutants")
            nh3 = st.number_input(
                "NH3 (Ammonia, µg/m³)",
                min_value=0.0, max_value=400.0,
                value=float(train_defaults.get("NH3", 16.0)),
                step=0.5,
                help="Ammonia emitted from agricultural and industrial activities."
            )
            co = st.number_input(
                "CO (Carbon Monoxide, mg/m³)",
                min_value=0.0, max_value=50.0,
                value=float(train_defaults.get("CO", 0.9)),
                step=0.1,
                help="Toxic gas produced by incomplete combustion of organic fuels."
            )
            so2 = st.number_input(
                "SO2 (Sulfur Dioxide, µg/m³)",
                min_value=0.0, max_value=300.0,
                value=float(train_defaults.get("SO2", 9.0)),
                step=0.5,
                help="Emitted by industrial fossil fuel combustion (power plants)."
            )
            o3 = st.number_input(
                "O3 (Ground-level Ozone, µg/m³)",
                min_value=0.0, max_value=300.0,
                value=float(train_defaults.get("O3", 31.0)),
                step=0.5,
                help="Secondary pollutant formed by photochemical reaction in sunlight."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Pollution-Risk Category", type="primary", use_container_width=True)

        if predict_btn:
            # Prepare input dictionary in exact model feature order
            user_input = {
                "PM2.5": pm25,
                "PM10": pm10,
                "NO2": no2,
                "NH3": nh3,
                "CO": co,
                "SO2": so2,
                "O3": o3
            }
            input_df = pd.DataFrame([user_input])[feature_cols]

            # Preprocess and Predict
            imputer = model_artifact["imputer"]
            model = model_artifact["model"]

            input_imputed = imputer.transform(input_df)
            predicted_category = model.predict(input_imputed)[0]
            
            # Predict probabilities if supported
            predicted_proba = None
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(input_imputed)[0]
                pred_idx = list(model.classes_).index(predicted_category)
                predicted_proba = probas[pred_idx] * 100

            category_details = CPCB_INFO.get(predicted_category, {
                "color": "#4a5568",
                "advisory": "No specific advisory available.",
                "caution": "Follow local air quality alerts."
            })

            # Display prediction banner
            st.markdown(
                f"""
                <div style="background-color: {category_details['color']}; color: white; padding: 24px; border-radius: 12px; text-align: center; margin: 20px 0;">
                    <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.9;">Predicted Risk Category</div>
                    <div style="font-size: 3rem; font-weight: 800; margin: 8px 0;">{predicted_category}</div>
                    {"<div style='font-size: 1rem; opacity: 0.95;'>Model Confidence: " + f"{predicted_proba:.1f}%</div>" if predicted_proba else ""}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Health Advisory Card
            st.markdown("### 📋 Health & Safety Advisory")
            st.write(f"**Health Impact:** {category_details['advisory']}")
            st.write(f"**Recommended Action:** {category_details['caution']}")

            # Disclaimer
            st.warning(
                "⚠️ **Disclaimer:** This classification is a statistical Machine Learning estimate based on input chemical "
                "concentrations. It is not an official real-time CPCB monitoring measurement and should not replace government health directives."
            )

# ---------------------------------------------------------
# SECTION B: POLLUTANT & CITY ANALYSIS
# ---------------------------------------------------------
elif section == "B. Pollutant & City Analysis":
    st.header("📊 Section B: Pollutant Relationships & Historical City Analysis")
    st.write(
        "Explores statistical relationships between chemical pollutants and numerical AQI, "
        "and identifies historical urban pollution hotspots using `data/city_day.csv`."
    )

    if df_history is None:
        st.error(f"⚠️ Dataset file not found at `{DATA_PATH}`.")
    else:
        # --- Part 1: Correlation Analysis ---
        st.subheader("1. Pollutant Correlations with Numerical AQI")
        st.write(
            "Pearson correlation coefficients between chemical pollutant concentrations and AQI "
            "across all valid historical records:"
        )

        pollutant_cols = ["PM2.5", "PM10", "NO2", "NH3", "CO", "SO2", "O3"]
        available_pollutants = [col for col in pollutant_cols if col in df_history.columns]

        if "AQI" in df_history.columns and available_pollutants:
            corr_df = df_history[available_pollutants + ["AQI"]].corr()["AQI"].drop("AQI").sort_values(ascending=True)

            fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
            bars = ax_corr.barh(corr_df.index, corr_df.values, color="#1f77b4", edgecolor="black", alpha=0.85)
            ax_corr.set_xlim(0, 1.0)
            ax_corr.set_xlabel("Pearson Correlation Coefficient (r) with AQI", fontweight="bold")
            ax_corr.set_title("Correlation of Pollutant Concentrations with Numerical AQI", fontweight="bold")
            ax_corr.grid(axis="x", linestyle="--", alpha=0.5)

            # Label values on bars
            for bar in bars:
                w = bar.get_width()
                ax_corr.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=9, fontweight="bold")

            st.pyplot(fig_corr)
            plt.close(fig_corr)

            strongest = corr_df.index[-1]
            second_strongest = corr_df.index[-2]
            st.info(
                f"**Key Findings:**\n"
                f"- **{strongest}** (r = {corr_df[strongest]:.2f}) and **{second_strongest}** (r = {corr_df[second_strongest]:.2f}) "
                f"exhibit the strongest positive correlations with AQI.\n"
                f"- In India's urban atmosphere, particulate matter routinely breaches standard thresholds, making it the primary driver of high AQI values.\n"
                f"- **Important Principle:** Correlation measures mathematical association and does not prove physical causality."
            )

        st.markdown("---")

        # --- Part 2: Historical City Analysis ---
        st.subheader("2. Historical City-Level AQI Comparison (2015–2020)")
        st.write(
            "Average AQI per city computed strictly on cities with **at least 100 valid daily observations**. "
            "*(Represents historical averages in the dataset, not current real-time air quality.)*"
        )

        city_stats = (
            df_history.dropna(subset=["AQI"])
            .groupby("City")["AQI"]
            .agg(Observations="count", Average_AQI="mean")
            .reset_index()
        )
        valid_cities = city_stats[city_stats["Observations"] >= 100].sort_values(by="Average_AQI", ascending=False)

        top_polluted = valid_cities.head(5).iloc[::-1]  # reverse for horizontal bar chart
        cleanest = valid_cities.tail(5).iloc[::-1]

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("#### Top 5 Highest Average AQI (Most Polluted)")
            fig_p, ax_p = plt.subplots(figsize=(6, 3.5))
            bars_p = ax_p.barh(top_polluted["City"], top_polluted["Average_AQI"], color="#d9534f", edgecolor="black", alpha=0.85)
            ax_p.set_xlabel("Historical Average AQI", fontweight="bold")
            ax_p.set_title("Cities with Highest Average AQI", fontweight="bold")
            ax_p.grid(axis="x", linestyle="--", alpha=0.5)
            for bar in bars_p:
                w = bar.get_width()
                ax_p.text(w + 5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", fontsize=9, fontweight="bold")
            st.pyplot(fig_p)
            plt.close(fig_p)

        with col_c2:
            st.markdown("#### Top 5 Lowest Average AQI (Cleanest)")
            fig_c, ax_c = plt.subplots(figsize=(6, 3.5))
            bars_c = ax_c.barh(cleanest["City"], cleanest["Average_AQI"], color="#5cb85c", edgecolor="black", alpha=0.85)
            ax_c.set_xlabel("Historical Average AQI", fontweight="bold")
            ax_c.set_title("Cities with Lowest Average AQI", fontweight="bold")
            ax_c.grid(axis="x", linestyle="--", alpha=0.5)
            for bar in bars_c:
                w = bar.get_width()
                ax_c.text(w + 3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", fontsize=9, fontweight="bold")
            st.pyplot(fig_c)
            plt.close(fig_c)

# ---------------------------------------------------------
# SECTION C: MODEL PERFORMANCE
# ---------------------------------------------------------
elif section == "C. Model Performance":
    st.header("📈 Section C: Model Evaluation & Performance")
    st.write(
        "Empirical classification metrics calculated strictly on the held-out "
        "**20% test dataset (4,970 samples)** that the Random Forest never saw during training."
    )

    if model_artifact is None:
        st.error("⚠️ Model artifact not found! Run `python train_model.py` to generate metrics.")
    else:
        metrics = model_artifact.get("metrics", {})
        classes = model_artifact.get("classes", [])
        cm = np.array(model_artifact.get("confusion_matrix", []))
        importances = model_artifact.get("feature_importances", {})

        # 4 Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.2f}%")
        m2.metric("Macro Precision", f"{metrics.get('precision_macro', 0.0) * 100:.2f}%")
        m3.metric("Macro Recall", f"{metrics.get('recall_macro', 0.0) * 100:.2f}%")
        m4.metric("Macro F1-Score", f"{metrics.get('f1_macro', 0.0) * 100:.2f}%")

        st.markdown("---")

        col_p1, col_p2 = st.columns([1.2, 1])

        with col_p1:
            st.subheader("Confusion Matrix (Test Partition)")
            if cm.size > 0 and classes:
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                cax = ax_cm.matshow(cm, cmap="Blues", alpha=0.85)
                fig_cm.colorbar(cax)

                ax_cm.set_xticks(range(len(classes)))
                ax_cm.set_yticks(range(len(classes)))
                ax_cm.set_xticklabels(classes, rotation=45, ha="left", fontsize=9)
                ax_cm.set_yticklabels(classes, fontsize=9)
                ax_cm.set_xlabel("Predicted Label", fontweight="bold")
                ax_cm.set_ylabel("True Label", fontweight="bold")

                # Annotate cells with counts
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        count = cm[i, j]
                        color = "white" if count > cm.max() / 2 else "black"
                        ax_cm.text(j, i, str(count), ha="center", va="center", color=color, fontweight="bold")

                st.pyplot(fig_cm)
                plt.close(fig_cm)
            else:
                st.info("Confusion matrix data unavailable.")

        with col_p2:
            st.subheader("Random Forest Feature Importances")
            if importances:
                feats = list(reversed(list(importances.keys())))
                vals = [importances[k] * 100 for k in feats]

                fig_fi, ax_fi = plt.subplots(figsize=(5, 4.5))
                bars_fi = ax_fi.barh(feats, vals, color="#2b5c8f", edgecolor="black", alpha=0.85)
                ax_fi.set_xlabel("Importance (%)", fontweight="bold")
                ax_fi.set_title("Relative Feature Contribution", fontweight="bold")
                ax_fi.grid(axis="x", linestyle="--", alpha=0.5)

                for bar in bars_fi:
                    w = bar.get_width()
                    ax_fi.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

                st.pyplot(fig_fi)
                plt.close(fig_fi)

        st.markdown("---")
        st.subheader("💡 Metric Definitions for Presentation & Viva")
        st.markdown("""
        - **Accuracy (80.97%):** Overall percentage of test samples whose pollution-risk category was correctly predicted.
        - **Precision (Macro 80.40%):** When the model predicts a specific risk tier, how often is it actually correct, averaged across all classes.
        - **Recall (Macro 77.26%):** Out of all actual occurrences of each risk tier, what proportion did the model correctly identify.
        - **Macro F1-Score (78.68%):** Harmonic mean of Precision and Recall calculated equally across all categories, preventing frequent classes (Moderate/Satisfactory) from masking minority classes (Good/Severe).
        """)
