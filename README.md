# Credit Card Fraud Detection Using Machine Learning

## About the Project

Credit card fraud is a major challenge for financial institutions and digital payment systems. Fraudulent transactions can cause financial losses and undermine customer trust.

This project develops a machine learning solution to identify potentially fraudulent credit card transactions. Using transaction data and classification algorithms, the project explores how data science can support fraud monitoring and risk management.

The project includes exploratory data analysis, data preprocessing, class imbalance handling, model training, evaluation, and the development of an interactive dashboard using Streamlit.

## Objectives

- Analyze credit card transaction data to identify patterns associated with fraudulent activity.
- Address class imbalance using the Synthetic Minority Over-sampling Technique (SMOTE).
- Train and compare Logistic Regression, Random Forest, and XGBoost classifiers.
- Evaluate model performance using precision, recall, F1-score, ROC-AUC, and Precision-Recall AUC.
- Adjust the classification threshold to balance fraud detection and false alerts.
- Develop an interactive Streamlit dashboard for transaction analysis and fraud prediction.

## Dataset

The project uses the European credit card transactions dataset available on Kaggle.

**Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains 284,807 transactions, including 492 fraudulent transactions. The target variable is `Class`, where:

- `0` represents a legitimate transaction.
- `1` represents a fraudulent transaction.

Most input variables (`V1` to `V28`) are anonymized numerical features. The dataset also includes `Time` and `Amount`.

## Methodology

1. **Exploratory Data Analysis:** Examine the dataset structure, missing values, duplicates, transaction amounts, and class distribution.
2. **Data Preprocessing:** Separate input features from the target variable and standardize `Time` and `Amount`.
3. **Class Imbalance Handling:** Apply SMOTE to the training data only.
4. **Model Training:** Train Logistic Regression, Random Forest, and XGBoost classifiers.
5. **Model Evaluation:** Compare precision, recall, F1-score, ROC-AUC, and Precision-Recall AUC.
6. **Threshold Optimization:** Adjust the Random Forest decision threshold to investigate the trade-off between missed fraud and false alerts.
7. **Dashboard Development:** Use Streamlit to display predictions, fraud probabilities, summary metrics, and downloadable results.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Matplotlib
- Streamlit
- Joblib
- Google Colab
- GitHub

## Model Results

The Random Forest model was selected for the dashboard. At a classification threshold of `0.75`, the reported evaluation results were:

| Metric | Result |
|---|---:|
| Precision | 95.00% |
| Recall | 77.55% |
| F1-score | 85.39% |
| True Positives | 76 |
| False Positives | 4 |
| False Negatives | 22 |
| True Negatives | 56,860 |

These results indicate that the model correctly identified 76 fraudulent transactions, while 22 fraudulent transactions were missed at the selected threshold.

**Important:** The threshold was selected after examining results on the test set. These figures should therefore be treated as preliminary rather than as a fully independent estimate of performance. A more rigorous evaluation should select the threshold on validation data and evaluate it once on an untouched test set.

## Streamlit Dashboard

The dashboard is designed to support:

- CSV transaction data uploads.
- Fraud probability estimation.
- Adjustable fraud alert thresholds.
- Summary metrics and transaction classifications.
- Visualization of fraud probabilities.
- Downloadable prediction results.
- Optional evaluation when genuine transaction labels are provided.

## Project Structure

```text
credit-card-fraud-detection/
├── app.py
├── requirements.txt
├── README.md
└── fraud_detection_artifacts/
    ├── random_forest_fraud_model.pkl
    ├── fraud_scaler.pkl
    └── fraud_model_config.pkl
```

## Installation and Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Ensure that the three model artifact files are available in the `fraud_detection_artifacts/` directory.

## Limitations

- The dataset contains anonymized features, limiting direct interpretation of individual variables.
- Fraud detection involves a highly imbalanced classification problem.
- Model performance may change when applied to transactions from different periods or institutions.
- A high prediction probability does not prove that a transaction is fraudulent.
- The model is a portfolio and educational project, not a validated production fraud detection system.

## Future Improvements

- Evaluate the model using a dedicated validation set and an untouched test set.
- Tune model hyperparameters and decision thresholds.
- Explore explainability techniques such as SHAP.
- Monitor model performance and potential data drift.
- Deploy the dashboard online for interactive demonstrations.

## Disclaimer

This project is for educational and portfolio purposes only. Predictions should not be used as the sole basis for real financial decisions.

## Author

Developed as a machine learning and data analytics portfolio project by LENY N. NAPALA.
