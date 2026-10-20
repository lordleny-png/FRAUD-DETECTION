
import joblib
import pandas as pd
import streamlit as st

from huggingface_hub import hf_hub_download
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Credit Card Fraud Detection")
st.caption(
    "Random Forest model | Machine Learning | "
    "Educational portfolio project"
)

# --------------------------------------------------
# LOAD MODEL FILES FROM HUGGING FACE
# --------------------------------------------------
REPO_ID = "Leny7/fraud-detection-model"

@st.cache_resource
def load_artifacts():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="random_forest_fraud_model.pkl"
    )

    scaler_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="fraud_scaler.pkl"
    )

    config_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="fraud_model_config.pkl"
    )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    config = joblib.load(config_path)

    return model, scaler, config


try:
    model, scaler, config = load_artifacts()
    st.success("Fraud detection model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------
default_features = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

feature_names = config.get("feature_names", default_features)
feature_names = [
    name for name in feature_names if name != "Class"
]

scaled_features = config.get(
    "scaled_features",
    ["Amount", "Time"]
)

threshold = float(config.get("threshold", 0.75))

st.metric("Fraud Alert Threshold", f"{threshold:.2f}")

# --------------------------------------------------
# CSV UPLOAD
# --------------------------------------------------
st.subheader("📂 Upload Transactions")

st.write(
    "Upload a CSV containing Time, V1–V28, and Amount. "
    "You can analyze one transaction or multiple transactions."
)

uploaded_file = st.file_uploader(
    "Choose your transaction CSV file",
    type=["csv"],
    help="Your CSV must contain the 30 required model input features."
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        st.subheader("Uploaded Data")
        st.write(f"Number of transactions: {len(df):,}")
        st.dataframe(df.head(10), use_container_width=True)

        # Check required columns
        missing = [
            col for col in feature_names
            if col not in df.columns
        ]

        if missing:
            st.error(
                "Your CSV is missing these required columns: "
                + ", ".join(missing)
            )
            st.stop()

        # Prepare the model input in the correct column order
        X = df[feature_names].copy()

        # Ensure features are numeric
        for col in feature_names:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if X.isna().any().any():
            st.error(
                "Some required features contain missing or "
                "non-numeric values. Please correct your CSV."
            )
            st.stop()

        # Apply the saved scaler only to the configured features
        if not all(col in X.columns for col in scaled_features):
            st.error(
                "The configured scaling features do not match "
                "the model input columns."
            )
            st.stop()

        X.loc[:, scaled_features] = scaler.transform(
            X[scaled_features]
        )

        # --------------------------------------------------
        # FRAUD PREDICTIONS
        # --------------------------------------------------
        if st.button("🔍 Analyze Transactions", type="primary"):
            probabilities = model.predict_proba(X)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            results = df.copy()
            results["Fraud_Probability"] = probabilities
            results["Fraud_Prediction"] = predictions
            results["Risk_Level"] = [
                "Potential Fraud" if p == 1 else "Likely Legitimate"
                for p in predictions
            ]

            fraud_count = int(predictions.sum())
            legitimate_count = len(predictions) - fraud_count

            st.subheader("📊 Analysis Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Transactions Analyzed",
                f"{len(results):,}"
            )
            col2.metric(
                "Potential Fraud Alerts",
                f"{fraud_count:,}"
            )
            col3.metric(
                "Likely Legitimate",
                f"{legitimate_count:,}"
            )

            st.subheader("Prediction Results")
            st.dataframe(
                results,
                use_container_width=True
            )

            # Display a simple chart
            st.subheader("Transaction Classification")

            chart_data = pd.DataFrame({
                "Category": [
                    "Likely Legitimate",
                    "Potential Fraud"
                ],
                "Transactions": [
                    legitimate_count,
                    fraud_count
                ]
            }).set_index("Category")

            st.bar_chart(chart_data)

            # Optional evaluation when true labels are supplied
            if "Class" in df.columns:
                st.subheader("Model Evaluation")

                y_true = pd.to_numeric(
                    df["Class"], errors="coerce"
                )

                if y_true.isna().any():
                    st.warning(
                        "The Class column contains invalid labels; "
                        "evaluation was skipped."
                    )
                else:
                    y_true = y_true.astype(int)

                    if not y_true.isin([0, 1]).all():
                        st.warning(
                            "The Class column must contain only 0 and 1."
                        )
                    else:
                        cm = confusion_matrix(
                            y_true, predictions, labels=[0, 1]
                        )

                        st.write(
                            "Confusion matrix "
                            "(rows = actual, columns = predicted):"
                        )

                        st.dataframe(
                            pd.DataFrame(
                                cm,
                                index=[
                                    "Actual Legitimate",
                                    "Actual Fraud"
                                ],
                                columns=[
                                    "Predicted Legitimate",
                                    "Predicted Fraud"
                                ]
                            )
                        )

                        st.text(
                            classification_report(
                                y_true,
                                predictions,
                                labels=[0, 1],
                                target_names=[
                                    "Legitimate",
                                    "Fraud"
                                ],
                                zero_division=0
                            )
                        )

            # Download predictions
            csv_output = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv_output,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing the CSV: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Educational project only. Predictions indicate potential "
    "fraud risk and should not be treated as definitive proof."
)
