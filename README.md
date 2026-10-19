# FRAUD-DETECTION
# Credit Card Fraud Detection

## Project Overview
This project uses machine learning to identify potentially fraudulent credit card transactions. It compares classification models and provides a Streamlit dashboard for transaction analysis.

## Objectives
- Detect potentially fraudulent transactions.
- Address class imbalance during model training.
- Evaluate performance using precision, recall, F1-score, ROC-AUC and Average Precision.
- Provide transaction-level and batch predictions through an interactive dashboard.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest
- SMOTE
- Matplotlib
- Streamlit

## Key Results
The preliminary Random Forest evaluation at a decision threshold of 0.75 produced:

- Precision: 95.00%
- Recall: 77.55%
- F1-score: 0.8539
- True positives: 76
- False positives: 4
- False negatives: 22
- True negatives: 56,860

These are preliminary test-set results because the threshold was selected after examining test-set performance. A final evaluation should select the threshold on validation data and evaluate it on an untouched test set.

## Dashboard Features
- CSV transaction upload
- Adjustable fraud decision threshold
- Fraud probability estimates
- Batch classification and summary charts
- Downloadable prediction results

## Run Locally
Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Input Data
The model expects `Time`, `V1` through `V28`, and `Amount`. The `V1`–`V28` columns are transformed features from the source dataset. The dashboard applies the saved scaler to `Time` and `Amount` only.

## Disclaimer
This is an educational portfolio project. Predictions are not definitive proof of fraud and should not be used as the sole basis for financial decisions.

