# 🌿 AirAware: Explainable Air Pollution Risk Analysis & Forecasting

**Academic Context:** B.Tech Computer Engineering — Machine Learning Mini Project  
**Core Technologies:** Python 3.14, Scikit-Learn, Pandas, NumPy, Matplotlib, Streamlit, Joblib  
**Architecture:** Pure Offline Traditional Machine Learning • Zero External APIs • Zero Databases  

---

## 📌 1. Project Overview & Distinguishing Objective

Most basic air quality projects are simple regression dashboards that try to predict an arbitrary numerical index. **AirAware** delivers three practical, beginner-friendly outcomes built around environmental risk:

1. **Pollution-Risk Classification:** Categorizes ambient air into official Central Pollution Control Board (CPCB) health risk tiers (**Good, Satisfactory, Moderate, Poor, Very Poor, Severe**) from chemical pollutant concentrations using a supervised **Random Forest Classifier**.
2. **Pollutant Relationship Analysis:** Examines statistical correlations between individual chemical concentrations ($PM_{2.5}, PM_{10}, CO, NO_2, SO_2, O_3, NH_3$) and AQI, highlighting which pollutants dominate air quality degradation while reinforcing that correlation is not causation.
3. **Historical City-Level Insights:** Aggregates empirical multi-year observations to benchmark historical urban pollution hotspots and identify comparatively cleaner airsheds across India.

---

## 🗄️ 2. Primary Dataset

- **File:** [`data/city_day.csv`](file:///c:/Users/Admin/Desktop/AIR%20QUALITY%20INDEX/data/city_day.csv)
- **Total Records:** 29,531 daily records across 26 major Indian cities (2015–2020)
- **Valid Labeled Samples:** 24,850 records (after excluding 4,681 records with unmonitored `AQI_Bucket`)
- **Input Features (7 Criteria Chemical Pollutants):**
  - $PM_{2.5}$ (Fine Particulate Matter, $\mu\text{g/m}^3$)
  - $PM_{10}$ (Coarse Particulate Matter, $\mu\text{g/m}^3$)
  - $NO_2$ (Nitrogen Dioxide, $\mu\text{g/m}^3$)
  - $NH_3$ (Ammonia, $\mu\text{g/m}^3$)
  - $CO$ (Carbon Monoxide, $\text{mg/m}^3$)
  - $SO_2$ (Sulfur Dioxide, $\mu\text{g/m}^3$)
  - $O_3$ (Ground-Level Ozone, $\mu\text{g/m}^3$)
- **Classification Target:** `AQI_Bucket` (6 discrete public health categories)

---

## 🛡️ 3. Preprocessing & Leakage Prevention Rules

To guarantee academic rigor and realistic evaluation:
1. **Target Derivation Exclusion:** Continuous `AQI` and the target `AQI_Bucket` are strictly excluded from the model feature matrix $X$. The classifier learns patterns solely from chemical concentrations.
2. **No Data Fabrication:** Records lacking ground-truth target labels (4,681 rows) are excluded rather than filled with synthetic or invented labels.
3. **Isolated Imputation:** Missing pollutant values are handled using median imputation (`SimpleImputer(strategy='median')`), fitted **strictly on `X_train`** and applied downstream to `X_test`. This prevents test-set distribution leakage into training.
4. **Stratified Split:** Data is partitioned into an **80% training set (19,880 samples)** and a **20% testing set (4,970 samples)** with `random_state=42`, preserving class proportions across minority and majority risk tiers.

---

## 🏆 4. Empirical Evaluation Results (Held-Out Test Data)

Evaluated on **4,970 unseen test samples**:

| Evaluation Metric | Test Score | Description |
|:---|:---:|:---|
| **Overall Accuracy** | **80.97%** | Percentage of test instances whose exact CPCB risk tier was predicted correctly |
| **Macro Precision** | **80.40%** | Average precision across all 6 classes (unweighted by class frequency) |
| **Macro Recall** | **77.26%** | Average sensitivity across all 6 classes |
| **Macro F1-Score** | **78.68%** | Harmonic mean of precision and recall treating all risk categories equally |

### Class-Wise Classification Breakdown:
| CPCB Risk Tier | Precision | Recall | F1-Score | Test Support |
|:---|:---:|:---:|:---:|:---:|
| **Good** | 0.7713 | 0.6418 | 0.7006 | 268 |
| **Satisfactory** | 0.8232 | 0.8492 | 0.8360 | 1,645 |
| **Moderate** | 0.8148 | 0.8420 | 0.8282 | 1,766 |
| **Poor** | 0.7351 | 0.6637 | 0.6975 | 556 |
| **Very Poor** | 0.8013 | 0.8030 | 0.8021 | 467 |
| **Severe** | 0.8784 | 0.8358 | 0.8566 | 268 |

### Random Forest Feature Importances:
1. **$PM_{2.5}$**: **31.95%** (Dominant fine respirable particles)
2. **$PM_{10}$**: **19.48%** (Coarse respirable dust)
3. **$CO$**: **15.94%** (Combustion byproduct)
4. **$O_3$**: **9.20%** (Photochemical oxidant)
5. **$NO_2$**: **9.05%** (Vehicular emissions)
6. **$SO_2$**: **8.14%** (Industrial emissions)
7. **$NH_3$**: **6.23%** (Agricultural & waste emissions)

*Particulate matter ($PM_{2.5}$ and $PM_{10}$) accounts for over 51% of total predictive weight.*

---

## 🏙️ 5. Pollutant Relationships & City Hotspot Findings

### A. Pollutant Correlations with AQI:
- **$PM_{10}$ ($r = 0.80$)**, **$CO$ ($r = 0.68$)**, and **$PM_{2.5}$ ($r = 0.66$)** demonstrate the highest positive correlation with AQI.
- *Scientific Note:* High correlation indicates statistical co-occurrence and shared emission sources, not direct single-variable causality.

### B. Historical City-Level Rankings ($\ge 100$ valid daily records):
- **Top 5 Historically Most Polluted Cities:**
  1. Ahmedabad (Avg AQI: 452.1)
  2. Delhi (Avg AQI: 259.5)
  3. Patna (Avg AQI: 240.8)
  4. Gurugram (Avg AQI: 225.1)
  5. Lucknow (Avg AQI: 218.0)
- **Top 5 Historically Cleanest Cities:**
  1. Aizawl (Avg AQI: 34.8)
  2. Shillong (Avg AQI: 53.8)
  3. Coimbatore (Avg AQI: 73.0)
  4. Thiruvananthapuram (Avg AQI: 75.9)
  5. Ernakulam (Avg AQI: 92.4)

*(Note: These figures reflect historical averages across the 2015–2020 dataset and do not represent real-time measurements).*

---

## 🖥️ 6. Streamlit Web Dashboard Structure

The web dashboard is organized into three clean sections:

1. **🧮 Section A: Pollution-Risk Prediction:**
   - Numerical input fields for the 7 criteria pollutants with pre-filled training medians.
   - Interactive prediction button that runs the pre-trained Random Forest model.
   - Visual CPCB color badge with standard health statement and actionable public health advisory.
   - Clear disclaimer that predictions are ML estimates, not official real-time CPCB readings.
2. **📊 Section B: Pollutant & City Analysis:**
   - Horizontal bar chart of pollutant correlations with AQI.
   - Comparative bar charts for the top 5 most polluted and top 5 cleanest Indian cities.
3. **📈 Section C: Model Performance:**
   - Test accuracy, precision, recall, and macro F1 metric cards.
   - $6 \times 6$ confusion matrix heatmap.
   - Relative feature importance breakdown and plain-language metric definitions for presentation.

---

## 📁 7. Project File Structure

```text
AIR QUALITY INDEX/
├── data/
│   └── city_day.csv                  # Primary historical Indian AQI dataset (2015–2020)
├── models/
│   └── pollution_classifier.joblib   # Serialized model, imputer, metrics & feature metadata
├── train_model.py                    # Standalone training script (data prep, split, train, eval, save)
├── app.py                            # Streamlit web application (3 clean sections)
├── requirements.txt                  # Minimal required Python dependencies
├── README.md                         # Project documentation and presentation guide
└── VIVA_QUESTIONS.md                 # 15 targeted Viva Voce questions & answers
```

---

## 💻 8. Quick Setup & Execution Instructions

### 1. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 2. Train and Evaluate the Classifier
```powershell
python train_model.py
```
*Outputs evaluation metrics on unseen test data and saves the artifact to `models/pollution_classifier.joblib`.*

### 3. Launch the Streamlit Web Application
```powershell
python -m streamlit run app.py
```
*Opens automatically in your default browser at `http://localhost:8501`.*
