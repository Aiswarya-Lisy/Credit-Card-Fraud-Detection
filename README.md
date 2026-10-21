# Credit Card Fraud Detection

Detecting fraudulent credit card transactions with anomaly detection. Fraud makes up under 0.4% of the data, so this project compares three anomaly detection models (Isolation Forest, One-Class SVM and an Autoencoder) and examines how well each one recovers the rare fraud cases.

## Problem Statement

Fraudulent transactions are extremely rare, which makes accuracy a misleading metric. This project builds models that flag unusual transactions, then evaluates them on precision, recall, F1-score and ROC-AUC for the fraud class.

## Dataset

- **Source:** Credit Card Transactions Fraud Detection Dataset on Kaggle (`fraudTest.csv`) 
- **Size:** 555,719 transactions
- **Fraud rate:** 0.39% (about 2,145 fraudulent transactions)
- **Features used:** merchant, category, amount, gender, customer and merchant coordinates, city population, plus time features (hour, day of week, month, day of month, time since first transaction)

The data file is not included in this repository because of its size. Download it from the link above and update the file path in the notebook.

## Approach

1. **Cleaning:** dropped identifier and personal columns (card number, name, street, city, state, zip, job, date of birth, transaction ID, unix time). No missing values were found.
2. **Feature engineering:** extracted hour, day of week, month, day of month and time since first transaction from the timestamp; label-encoded gender, merchant and category.
3. **EDA:** class distribution, transaction time patterns, feature histograms, correlation heatmaps and a t-SNE projection of a 1,000-transaction sample.
4. **Preprocessing:** standardised features with `StandardScaler`; stratified 70/30 train/test split; applied **SMOTE** to the training set to balance the classes (1,501 fraud vs 387,502 normal before SMOTE).
5. **Models:**
   - **Isolation Forest:** 100 trees, trained on the SMOTE-balanced training set.
   - **One-Class SVM:** RBF kernel, `nu=0.1`, trained on a 1% stratified subsample because of memory limits.
   - **Autoencoder (TensorFlow/Keras):** dense network 64-32-16-32-64, Adam optimiser, MSE loss, 50 epochs. A transaction is flagged as fraud if its reconstruction error is in the top 5%.
6. **Evaluation:** confusion matrix, precision, recall, F1-score and ROC-AUC for the fraud class.

## Results

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.0043 | 0.399 | 0.0086 | 0.545 |
| Autoencoder | 0.0152 | 0.197 | 0.0283 | 0.574 |
| One-Class SVM* | 0.0233 | 0.500 | 0.0444 | 0.696 |

Isolation Forest and Autoencoder were evaluated on the full test set (166,716 transactions, 644 fraud cases).

\*One-Class SVM was trained and tested on a 1% subsample, so its test set contained only 4 fraud cases. Its numbers are not comparable with the other two models.

## Key Findings

- All three anomaly detection models performed poorly on this dataset: precision stayed below 2.5% and ROC-AUC stayed close to 0.5-0.6 on the full test set.
- Isolation Forest caught the most fraud (about 40% recall) but flagged roughly 59,000 normal transactions as fraud. The Autoencoder had better precision and F1 but caught only about 20% of fraud.
- Accuracy looks high for every model (64% to 95%) but is meaningless here, because almost every transaction is legitimate.
- Fraud in this dataset is not well separated from normal activity in the raw feature space, so methods that model "normal" behaviour struggle.

## Limitations and Next Steps

- ROC-AUC for the Autoencoder and One-Class SVM was computed from binary predictions rather than continuous anomaly scores, which understates their ranking ability.
- The Autoencoder was trained on all transactions, including fraud, rather than on normal transactions only.
- Next step: add supervised classifiers (Random Forest, XGBoost) with SMOTE or class weights, and compare them against the anomaly detection models on the same test set.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE), TensorFlow/Keras, Matplotlib, Seaborn, Jupyter Notebook

## Project Structure

Credit-Card-Fraud-Detection/
├── credit.ipynb   # EDA, preprocessing, models, evaluation
├── requirements.txt
└── README.md

## How to Run
git clone https://github.com/Aiswarya-Lisy/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
# download fraudTest.csv (see Dataset) and update the path in the notebook
jupyter notebook
