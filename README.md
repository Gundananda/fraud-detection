# 💳 Fraud Detection with SMOTE & Random Forest (Scikit‑learn)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imblearn-0.10%2B-7B1FA2)](https://imbalanced-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A lightweight end‑to‑end ML pipeline for classifying fraudulent bank transactions with feature engineering, class imbalance handling (SMOTE), and a Random Forest classifier.

</div>

---

## 📌 Overview
This project builds a transaction fraud detector using classic ML. It:
- Cleans and engineers features from raw transactions
- Addresses extreme class imbalance with SMOTE
- Trains a tuned Random Forest
- Evaluates with ROC‑AUC, confusion matrix, and stratified 5‑fold CV
- Surfaces actionable business insights and prevention strategies

> ⚠️ Disclaimer: Educational & research purposes only. Always validate with domain experts before deploying to production.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| ⚖️ Class Imbalance Handling | SMOTE upsampling on the training set to balance the rare fraud class. |
| 🧮 Feature Engineering | errorBalanceOrig and errorBalanceDest to capture balance anomalies. |
| 🌲 Robust Model | RandomForestClassifier (n_estimators=200, class_weight='balanced'). |
| 📊 Evaluation Suite | Classification report, ROC‑AUC, confusion matrix, calibration-ready probs. |
| 🔁 Stratified 5‑Fold CV | cross_val_score with ROC‑AUC for stability checks. |
| 🧭 Diagnostics | Correlation heatmap for multicollinearity, feature importance ranking. |

---

## 📂 Project Structure

```plaintext
fraud-detection-rf/
├── fraud_detection.ipynb       # Main notebook
├── README.md                   # This file
├── LICENSE                     # MIT
└── imgs/                       # Plots (add your saved figures here)
    ├── corr_heatmap.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

---

## 📦 Dataset

- File used in notebook: Fraud.csv (500MB; not stored in repo due to size)
- Columns used: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, nameOrig, nameDest, isFraud, isFlaggedFraud
- After cleaning:
  - Categorical encoding: LabelEncoder on type
  - Engineered: errorBalanceOrig = newbalanceOrig + amount − oldbalanceOrg
  - Engineered: errorBalanceDest = oldbalanceDest + amount − newbalanceDest
  - Dropped: nameOrig, nameDest (IDs)

Note: Place the dataset at the path you set in the notebook (e.g., /content/Fraud.csv or data/Fraud.csv).

---

## 🧠 Technical Details

- Algorithm: RandomForestClassifier
  - n_estimators=200, class_weight='balanced', random_state=42
- Sampling: SMOTE on training split only
- Split: train/test = 80/20 (stratified)
- CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Metrics: Accuracy, ROC‑AUC, classification report, confusion matrix
- Libraries: pandas, numpy, scikit‑learn, imbalanced‑learn, seaborn, matplotlib

Class imbalance:
- Before SMOTE (train): 0 → 5,083,526; 1 → 6,570
- After SMOTE (train): 0 → 5,083,526; 1 → 5,083,526

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Notebook/Lab

### Installation
```bash
# Clone your repo
git clone https://github.com/Gundananda/fraud-detection.git
cd fraud-detection

# Create env (optional)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install deps
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

### Run
```bash
# Launch the notebook
jupyter notebook fraud_detection.ipynb
```

Update file_path inside the notebook to point to your Fraud.csv and run all cells.

---

## 📊 Results & Visualizations

- Test set metrics:
  - Accuracy: 1.0000
  - ROC‑AUC: 0.9994
  - Classification report (test):
    - Class 0 (Non‑Fraud): precision=1.00, recall=1.00, f1=1.00 (support=1,270,881)
    - Class 1 (Fraud): precision=0.98, recall=1.00, f1=0.99 (support=1,643)

- Confusion matrix (test):
  - TN=1,270,847, FP=34, FN=4, TP=1,639

- 5‑fold Stratified CV (ROC‑AUC):
  - Scores: [0.9991, 0.9966, 0.9994, 0.9982, 0.9982]
  - Mean: 0.9983

- Feature importance (top):
  - errorBalanceOrig: 0.4277
  - newbalanceOrig: 0.1826
  - oldbalanceOrg: 0.1363
  - amount: 0.0837
  - type: 0.0650
  - errorBalanceDest: 0.0397
  - step: 0.0288
  - newbalanceDest: 0.0264
  - oldbalanceDest: 0.0090
  - isFlaggedFraud: 0.0008

<div align="center">

Add your saved figures here (place under imgs/ and update paths if needed):

<img src="imgs/confusion_matrix.png" width="520"/>
<img src="imgs/roc_curve.png" width="520"/>
<img src="imgs/corr_heatmap.png" width="700"/>
<img src="imgs/feature_importance.png" width="600"/>

</div>

Note: CV shown here evaluates the model with class_weight on the original imbalanced data. To apply SMOTE within CV without leakage, wrap SMOTE + model in an imblearn Pipeline and use cross_val_score on that pipeline.

---

## 🔎 Insights & Business Actions

- Key fraud indicators
  - Transaction type: TRANSFER and CASH‑OUT are strongly associated with fraud.
  - Large amounts carry higher risk.
  - Balance anomalies (errorBalanceOrig, errorBalanceDest) signal manipulation.

- Why these make sense
  - Fraud flows often chain TRANSFER → CASH‑OUT to quickly move and withdraw funds.
  - High‑value targets maximize returns for fraudsters.
  - Inconsistent balance updates hint at account tampering and synthetic behaviors.

- Prevention strategies
  - Flag high‑value TRANSFER/CASH‑OUT sequences for review or step‑up auth.
  - Real‑time anomaly detection on balance deltas and velocity checks.
  - MFA for risky transfers; per‑day limits and dynamic risk thresholds.

- Validate impact
  - Compare detection/recall before vs. after policies.
  - Track change in monetary loss, false positives, review queue size, and latency.

---

## 🧪 Reproducibility Tips

- Set random_state=42 consistently (split, SMOTE, model).
- Log data version and preprocessing config.
- Save model and encoders:
```python
import joblib
joblib.dump(rf_model, "models/rf_model.joblib")
joblib.dump(le, "models/type_label_encoder.joblib")
```

---

## 📄 License
This project is released under the MIT License. See LICENSE for details.

---


⭐️ If this helped, a star would be awesome!
