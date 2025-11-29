# ⭐ Fraud Detection using Machine Learning (Random Forest + SMOTE)
*Python • scikit-learn • Imbalanced-Learn*

A lightweight, end-to-end machine learning pipeline for detecting fraudulent financial transactions using classical ML, feature engineering, and imbalance-handling techniques.

---

## 🔍 Overview

This project implements a **Random Forest–based fraud detection system** trained on a large financial transactions dataset (~500MB). It applies:

- **Feature engineering**
- **SMOTE oversampling**
- **Random Forest Classifier**
- **Evaluation with ROC–AUC, Accuracy, Confusion Matrix, CV**

> ⚠️ **Note:** The dataset is NOT included (500MB).  
> Add it manually to:  
> `data/Fraud.csv`

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
│── data/ # Place Fraud.csv here (ignored by Git)
│── models/ # Optional saved models
│── notebooks/
│ └── fraud_detection.ipynb # Main notebook
│── README.md
│── requirements.txt
│── LICENSE

