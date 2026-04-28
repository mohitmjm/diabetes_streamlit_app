# 🩺 DiabetesAI Pro — Advanced Diabetes Prediction App

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit)](https://diabetesappapp-ac8kqdm6bbm77gwikmt2kv.streamlit.app/)

A production-grade machine learning web application that predicts diabetes risk using a **Stacking Ensemble of 6 models** combined with SMOTE oversampling, advanced feature engineering, and a personalised health summary report.

---

## 🚀 Live Demo

**🌐 Deployed App:** [https://diabetesappapp-ac8kqdm6bbm77gwikmt2kv.streamlit.app/](https://diabetesappapp-ac8kqdm6bbm77gwikmt2kv.streamlit.app/)

> Or run locally with `streamlit run app.py` → open [http://localhost:8501](http://localhost:8501)

---

## ✨ Features

- 🔮 **Risk Gauge** — Interactive dial showing diabetes probability score (0–100%)
- 📝 **Personalised Health Summary** — Auto-generated findings & recommendations per patient
- 📡 **Radar Chart** — Visual breakdown of all 6 risk factors
- 📊 **Data Explorer** — Feature distributions, correlation heatmap
- 📈 **Model Performance Tab** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Feature Importance
- 🎨 **Premium Dark UI** — Glassmorphism design with smooth gradients

---

## 🤖 ML Architecture

### Stacking Ensemble (6 Base Models + Meta-Learner)

| Layer | Model | Purpose |
|---|---|---|
| Base | 🌲 Random Forest (500 trees) | Bagging, robust to noise |
| Base | 🌳 Extra Trees (400 trees) | High variance capture |
| Base | 📈 Gradient Boosting (300 est.) | Sequential error correction |
| Base | ⚡ XGBoost (400 est., regularized) | Best single-model performance |
| Base | 🔮 SVM (RBF kernel, C=3) | Margin maximisation |
| Base | 📍 KNN (k=7, distance-weighted) | Local pattern matching |
| Meta | 🧠 Logistic Regression | Optimal combination of base models |

### Advanced Techniques

| Technique | Description |
|---|---|
| **SMOTE** | Synthesises new minority-class (diabetic) samples to fix class imbalance |
| **RobustScaler** | IQR-based scaling — resistant to outliers |
| **EllipticEnvelope** | Removes statistical outliers via Mahalanobis distance |
| **Feature Engineering** | 6 derived features: Glucose×BMI, Age×Pregnancies, Insulin/Glucose ratio, BMI×Age, Glucose², Risk Score |
| **5-Fold Stratified CV** | Unbiased, reliable accuracy estimation |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ✅ Test Accuracy | **~85–88%** |
| 🎯 Precision | ~84% |
| 🔁 Recall | ~83% |
| 📐 F1 Score | ~84% |
| 📉 ROC-AUC | ~91% |

> **Note:** The Pima Indians Diabetes dataset has a theoretical accuracy ceiling of ~83–88% due to its size (768 rows) and inherent label noise. Any model claiming 95%+ on this dataset is overfitting.

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/mohitmjm/diabetes_streamlit_app.git
cd diabetes_streamlit_app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
xgboost
plotly
joblib
imbalanced-learn
```

---

## 📁 Project Structure

```
diabetes_streamlit_app/
│
├── app.py              # Main Streamlit application
├── diabetes.csv        # Pima Indians Diabetes Dataset
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

---

## 🔬 Data Preprocessing Pipeline

1. **Zero-value imputation** — Biologically impossible zeros in `Glucose`, `BMI`, `BloodPressure`, `SkinThickness`, `Insulin` are replaced with **class-stratified medians**
2. **Outlier removal** — `EllipticEnvelope` detects and removes ~5% statistical outliers
3. **Feature engineering** — 6 new medically-motivated features are derived
4. **SMOTE** — Training set is oversampled to create a perfectly balanced class distribution
5. **RobustScaler** — All features scaled using interquartile range

---

## 📝 Health Summary Feature

After prediction, the app generates a **personalised health report** containing:

- 🔴🟡🟢 Color-coded assessment of each parameter (Glucose, BMI, Blood Pressure, Insulin, Family History, Age)
- 💊 Specific, actionable health recommendations based on abnormal values
- 📡 Radar chart visualising all risk factors simultaneously
- 🏷️ Overall risk verdict: **Low / Moderate / High / Critical**

---

## 📊 Dataset

**Pima Indians Diabetes Database**
- Source: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- Records: 768 patients
- Features: 8 clinical measurements
- Target: Binary (Diabetic / Not Diabetic)

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It is **not a substitute** for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider.

---

## 👨‍💻 Author

**Mohit** — [@mohitmjm](https://github.com/mohitmjm)
