⭐ Fraud Detection using Machine Learning (Random Forest + SMOTE)

Python • scikit-learn • Imbalanced-Learn

A lightweight, end-to-end machine learning pipeline for detecting fraudulent financial transactions using classical ML, feature engineering, and imbalance-handling techniques.

🔍 Overview

This project implements a Random Forest–based fraud detection system trained on a large financial transactions dataset (~500MB). It applies:

Feature engineering

SMOTE oversampling

Random Forest Classifier

Evaluation with ROC–AUC, Accuracy, Confusion Matrix, and Cross-Validation

⚠️ The dataset is NOT included (500MB).
Add it manually to: data/Fraud.csv

✨ Key Features
Feature	Description
🧠 ML Model	RandomForestClassifier with class balancing
⚖️ Imbalance Handling	SMOTE oversampling for minority fraud class
🧹 Feature Engineering	Encoded transaction types + balance error fields
📊 Evaluation	ROC–AUC, Precision/Recall, Confusion Matrix, CV
🔍 Interpretability	Feature importance ranking
📂 Project Structure

fraud-detection/
│── data/ (Place Fraud.csv here, ignored by Git)
│── models/ (Optional saved models)
│── notebooks/
│ └── fraud_detection.ipynb
│── README.md
│── requirements.txt
│── LICENSE

📦 Dataset

This project uses a large anonymized financial transactions dataset.
Due to GitHub limitations, the dataset must be added manually:

data/Fraud.csv

Dataset fields include transaction type, amount, old/new balances, destination/origin accounts, and the fraud label isFraud.

🧠 Technical Details
Component	Description
📐 Model	Random Forest (200 estimators, balanced weights)
🧰 Frameworks	scikit-learn, pandas, numpy, seaborn, imbalanced-learn
⚡ Strategy	SMOTE oversampling + stratified train-test split
📏 Metrics	Accuracy, ROC–AUC, F1, Precision-Recall, Confusion Matrix
📊 Visualization	ROC curve, Confusion Matrix, Feature Importance
🚀 Getting Started

1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Add dataset
Place Fraud.csv in the data/ folder.

3️⃣ Open the notebook
Open and run: notebooks/fraud_detection.ipynb

📊 Results & Insights
🔹 Key Fraud Indicators

TRANSFER & CASH-OUT transaction types strongly correlate with fraud

Large transaction amounts increase risk

Balance inconsistencies (errorBalanceOrig, errorBalanceDest) reveal manipulation

🔹 Model Outputs

ROC–AUC Score

Confusion Matrix

Precision & Recall

5-fold Cross-Validation AUC

These confirm strong generalization and fraud-pattern capture.

🛡 Suggested Prevention Strategies

Flag rapid TRANSFER → CASH-OUT patterns

Use velocity checks for suspicious movement

Enforce multi-factor authentication for high-value transfers

Monitor abnormal balance updates using anomaly detection

📄 License

This project is licensed under the MIT License.
See the LICENSE file for full details.

<p align="center"><em>Simple. Effective. Interpretable Fraud Detection.</em></p>
