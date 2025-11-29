# ⭐ Fraud Detection using Machine Learning (Random Forest + SMOTE)
*Python • scikit-learn • Imbalanced-Learn*

A lightweight, end-to-end machine learning pipeline for detecting fraudulent financial transactions using classical ML, feature engineering, and imbalance-handling techniques.

---

## 🔍 Overview

This project implements a Random Forest–based fraud detection system trained on a large financial transactions dataset (~500MB). It includes:

- Feature engineering  
- SMOTE oversampling  
- Random Forest Classifier  
- Evaluation with ROC–AUC, Accuracy, Confusion Matrix, and Cross-Validation  

⚠️ Dataset NOT included (500MB).  
Add it manually to: `data/Fraud.csv`

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| 🧠 ML Model | RandomForestClassifier with class balancing |
| ⚖️ Imbalance Handling | SMOTE oversampling for minority fraud class |
| 🧹 Feature Engineering | Encoded transaction types + balance error fields |
| 📊 Evaluation | ROC–AUC, Precision/Recall, Confusion Matrix, CV |
| 🔍 Interpretability | Feature importance ranking |

---

## 📂 Project Structure

fraud-detection/  
│── data/ (Place Fraud.csv here — ignored by Git)  
│── models/ (Optional saved models)  
│── notebooks/  
│   └── fraud_detection.ipynb  
│── README.md  
│── requirements.txt  
│── LICENSE  

---

## 📦 Dataset

This project uses a large anonymized financial transactions dataset.  
Due to GitHub’s size limits, add the dataset manually:

data/Fraud.csv

Includes: transaction type, amount, old/new balances, destination/origin accounts, and fraud label `isFraud`.

---

## 🧠 Technical Details

| Component | Description |
|----------|-------------|
| 📐 Model | Random Forest (200 estimators, balanced weights) |
| 🧰 Frameworks | scikit-learn, pandas, numpy, seaborn, imbalanced-learn |
| ⚡ Strategy | SMOTE oversampling + stratified train-test split |
| 📏 Metrics | Accuracy, ROC–AUC, F1-score, Precision/Recall |
| 📊 Visualization | Confusion Matrix, ROC Curve, Feature Importance |

---

## 🚀 Getting Started

### 1️⃣ Install dependencies  
pip install -r requirements.txt

### 2️⃣ Add dataset  
Place the CSV file at:  
data/Fraud.csv

### 3️⃣ Run the notebook  
Open: notebooks/fraud_detection.ipynb

---

## 📊 Results & Insights

### 🔹 Key Fraud Indicators
- TRANSFER & CASH-OUT transaction types correlate strongly with fraud  
- High transaction amounts increase fraud risk  
- Balance inconsistencies (`errorBalanceOrig`, `errorBalanceDest`) reveal manipulation  

### 🔹 Model Outputs
- ROC–AUC score  
- Confusion Matrix  
- Precision & Recall metrics  
- 5-fold Cross-Validation AUC  

These confirm strong generalization and pattern recognition.

---

## 🛡 Recommended Prevention Measures

- Flag rapid **TRANSFER → CASH-OUT** sequences  
- Apply velocity checks on abnormal movement  
- Enforce MFA for high-value transfers  
- Use anomaly detection for balance inconsistencies  

---

## 📄 License

This project is licensed under the MIT License.  
See the LICENSE file for details.

---

<p align="center"><em>Simple. Effective. Interpretable Fraud Detection.</em></p>
