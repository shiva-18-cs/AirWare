# 🎓 AirAware: College Mini Project Viva Voce Q&A Guide

**Project Title:** AirAware: Explainable Air Pollution Risk Analysis & Forecasting  
**Domain:** B.Tech Computer Engineering — Machine Learning Mini Project  
**Target:** Final Semester / Technical Viva Voce Examination  

---

### Q1: What is the primary objective of the AirAware project?
**Answer:**  
AirAware is a traditional Machine Learning mini-project designed around three practical, interpretable objectives:
1. **Pollution-Risk Classification:** Predicting official Central Pollution Control Board (CPCB) health risk tiers (`Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, `Severe`) from ambient chemical pollutant concentrations.
2. **Pollutant Relationship Analysis:** Calculating statistical correlations between criteria pollutants and numerical AQI to explain which pollutants drive pollution severity.
3. **Historical City-Level Insights:** Benchmarking historical urban pollution hotspots and comparatively clean airsheds using multi-year empirical records.

---

### Q2: What are the CPCB pollution-risk categories and breakpoints?
**Answer:**  
The Central Pollution Control Board (CPCB) of India defines 6 standardized risk tiers:
- **Good (0 – 50):** Minimal impact.
- **Satisfactory (51 – 100):** Minor breathing discomfort to sensitive individuals.
- **Moderate (101 – 200):** Breathing discomfort to people with lung disease, asthma, and heart conditions.
- **Poor (201 – 300):** Breathing discomfort to most people on prolonged exposure.
- **Very Poor (301 – 400):** Respiratory illness to people on prolonged exposure.
- **Severe (401 – 500+):** Emergency impact affecting healthy adults and severely impacting vulnerable groups.

---

### Q3: Why is this framed primarily as a Classification task rather than continuous Regression?
**Answer:**  
In environmental public health, continuous numbers like 198 vs. 202 are practically indistinguishable, but crossing the threshold from *Moderate* (101–200) into *Poor* (201–300) triggers statutory health advisories, school closures, and emergency action plans (like GRAP in Delhi). Direct risk classification predicts the actionable decision tier that directly affects citizens and policy makers.

---

### Q4: Which input features are used, and which features were excluded to prevent Data Leakage?
**Answer:**  
- **Features Used (7 Criteria Pollutants):** `PM2.5`, `PM10`, `NO2`, `NH3`, `CO`, `SO2`, and `O3`.
- **Excluded Features:** The continuous `AQI` column and the target `AQI_Bucket` column were strictly excluded from the feature matrix $X$.
- **Why:** In India, `AQI_Bucket` is mathematically derived from `AQI`. If `AQI` were included in the features, the model would achieve 100% artificial accuracy by trivial memorization—a classic form of target leakage.

---

### Q5: How did you handle missing values in the dataset?
**Answer:**  
1. **Target Column (`AQI_Bucket`):** 4,681 rows with missing labels were dropped. Imputing or inventing labels would create artificial training targets.
2. **Feature Columns (Pollutants):** Missing pollutant readings were imputed using **median imputation** (`SimpleImputer(strategy='median')`). Median imputation is robust against extreme outliers (such as winter smog spikes).
3. **Leakage Prevention:** The imputer was fitted **strictly on `X_train`** and only applied to transform `X_test` and new user inputs in the web interface.

---

### Q6: What train-test split strategy did you use and why?
**Answer:**  
We used an **80:20 stratified split** with a fixed `random_state=42`:
- **Training Set:** 19,880 samples (80%)
- **Testing Set:** 4,970 samples (20%)
- **Why Stratified:** The classes have natural imbalance (Moderate: ~35.5%, Satisfactory: ~33.1%, vs. Good: ~5.4%, Severe: ~5.4%). Stratification guarantees that the minority classes are represented in the exact same proportion in both training and test sets.

---

### Q7: Why choose Random Forest Classifier as the algorithm?
**Answer:**  
1. **Handles Non-Linear Relationships:** Atmospheric pollutant chemistry is non-linear and governed by complex threshold effects. Decision trees naturally capture step-function thresholds.
2. **Bagging (Ensemble Averaging):** Random Forest aggregates predictions from 100 decision trees trained on bootstrap samples with random feature subsets, reducing variance and overfitting.
3. **Robust to Outliers:** Tree splits depend on ordering rather than distance metrics, making them resilient to extreme smog readings.
4. **Built-in Interpretability:** Provides feature importance scores without requiring complex external explainability toolkits.

---

### Q8: What were your actual model evaluation results?
**Answer:**  
On the held-out test partition of 4,970 samples:
- **Accuracy:** **80.97%**
- **Macro Precision:** **80.40%**
- **Macro Recall:** **77.26%**
- **Macro F1-Score:** **78.68%**
- Top performing classes were *Severe* (F1 = 0.8566), *Satisfactory* (F1 = 0.8360), and *Moderate* (F1 = 0.8282).

---

### Q9: Why is Macro F1-score more informative than simple Accuracy here?
**Answer:**  
Because of class imbalance:
- Moderate and Satisfactory account for ~68.6% of the data. A naive model that only ever predicted Moderate or Satisfactory could score ~68% accuracy while being completely useless for detecting *Good* or *Severe* days.
- **Macro F1-score** computes the harmonic mean of precision and recall for each of the 6 classes independently and averages them with equal weight. Achieving a **78.68% Macro F1** proves the model performs well even on rare, high-severity hazard events.

---

### Q10: What did the Confusion Matrix reveal about model misclassifications?
**Answer:**  
The confusion matrix revealed that almost all misclassifications occur between **adjacent severity categories** (e.g., Moderate misclassified as Satisfactory, or Poor misclassified as Very Poor). The model virtually never makes catastrophic errors (such as confusing a Good day with a Severe day).

---

### Q11: Which pollutants have the highest feature importance in your Random Forest?
**Answer:**  
- **$PM_{2.5}$:** **31.95%**
- **$PM_{10}$:** **19.48%**
- **$CO$:** **15.94%**
Together, particulate matter ($PM_{2.5} + PM_{10}$) accounts for over **51.4%** of the total decision-making importance. This aligns directly with physical reality: particulate levels routinely exceed safe limits by factors of 3x to 10x in Indian cities.

---

### Q12: What does the correlation analysis show, and why is "correlation not causation" emphasized?
**Answer:**  
- **Findings:** $PM_{10}$ ($r = 0.80$), $CO$ ($r = 0.68$), and $PM_{2.5}$ ($r = 0.66$) have the strongest positive correlation with AQI.
- **Correlation $\neq$ Causation:** High correlation indicates statistical co-occurrence, often because common sources (vehicular exhaust, biomass burning, industrial combustion) emit multiple pollutants simultaneously under stagnant weather conditions. It does not mean one pollutant caused the presence of another.

---

### Q13: What were the key findings from the historical city-level analysis?
**Answer:**  
For Indian cities with $\ge 100$ valid daily observations:
- **Top 5 Historically Most Polluted:** Ahmedabad (Avg AQI: 452.1), Delhi (259.5), Patna (240.8), Gurugram (225.1), Lucknow (218.0) — all situated in the Indo-Gangetic Plain subject to winter temperature inversions.
- **Top 5 Historically Cleanest:** Aizawl (Avg AQI: 34.8), Shillong (53.8), Coimbatore (73.0), Thiruvananthapuram (75.9), Ernakulam (92.4) — located in hilly or coastal terrains with higher ventilation and sea breezes.

---

### Q14: Why is the model persisted using Joblib instead of retraining in Streamlit?
**Answer:**  
1. **Computational Efficiency:** Training on 20,000 samples takes several seconds of CPU time and substantial memory. Re-training on every user slider interaction would make the UI laggy and unresponsive.
2. **Reproducibility:** A saved `.joblib` artifact packages the frozen model weights, fitted imputer, and verified test metrics, ensuring consistent, instantaneous inference in milliseconds.

---

### Q15: What are the main limitations and potential future improvements?
**Answer:**  
- **Limitations:** The model relies on chemical concentrations; it does not directly ingest local meteorological factors (such as wind speed or planetary boundary layer height) because matching microclimate sensors were unavailable in the historical dataset.
- **Future Improvements:** Integrating local micro-weather telemetry, deploying real-time low-cost IoT sensor networks, and experimenting with temporal sequence models (such as LSTMs) when continuous minute-by-minute streaming data becomes available.
