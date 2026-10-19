
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Credit Card Fraud Detection")
st.caption("Random Forest model — educational portfolio project")

# Load model artifacts
ARTIFACT_DIR = Path("fraud_detection_artifacts")

@st.cache_resource
def load_artifacts():
    model = joblib.load(
        ARTIFACT_DIR / "random_forest_fraud_model.pkl"
    )
    scaler = joblib.load(
        ARTIFACT_DIR / "fraud_scaler.pkl"
    )
    config = joblib.load(
        ARTIFACT_DIR / "fraud_model_config.pkl"
    )
    return model, scaler, config

try:
    model, scaler, config = load_artifacts()
except Exception as exc:
    st.error(
        "Model files could not be loaded. Check that all three "
        "PKL files are inside fraud_detection_artifacts."
    )
    st.exception(exc)
    st.stop()

feature_names = config["feature_names"]
scaled_features = config.get(
    "scaled_features", ["Amount", "Time"]
)
default_threshold = float(config.get("threshold", 0.75))

threshold = st.sidebar.slider(
    "Fraud decision threshold",
    min_value=0.05,
    max_value=0.99,
    value=min(max(default_threshold, 0.05), 0.99),
    step=0.01
)

st.sidebar.write(f"Current threshold: **{threshold:.2f}**")

st.subheader("Upload transaction data")
st.write(
    "Upload a CSV containing Time, V1–V28 and Amount. "
    "The Class column is optional and is used only for evaluation."
)

uploaded_file = st.file_uploader(
    "Choose a transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if df.empty:
        st.error("The uploaded file contains no transactions.")
        st.stop()

    # Keep labels separate from the model input
    actual_labels = df["Class"].copy() if "Class" in df.columns else None
    input_df = df.drop(columns=["Class"], errors="ignore")

    missing = [
        col for col in feature_names
        if col not in input_df.columns
    ]

    if missing:
        st.error(
            "Missing required features: " + ", ".join(missing)
        )
        st.stop()

    # Match the exact training feature order
    X = input_df[feature_names].copy()

    for col in feature_names:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isna().any().any():
        st.error(
            "Missing or non-numeric values were found. "
            "Please clean the CSV and upload it again."
        )
        st.stop()

    # Scale only the features scaled during training
    X[scaled_features] = scaler.transform(
        X[scaled_features]
    )

    # Generate probabilities and predictions
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = input_df.copy()
    results["Fraud_Probability"] = probabilities
    results["Predicted_Class"] = predictions
    results["Prediction"] = results["Predicted_Class"].map({
        0: "Legitimate",
        1: "Potential Fraud"
    })

    # Summary cards
    total = len(results)
    fraud_count = int(predictions.sum())
    legitimate_count = total - fraud_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions analyzed", f"{total:,}")
    c2.metric("Potential fraud alerts", f"{fraud_count:,}")
    c3.metric("Not flagged", f"{legitimate_count:,}")

    st.subheader("Prediction results")

    display_columns = [
        c for c in [
            "Time", "Amount", "Fraud_Probability",
            "Prediction", "Predicted_Class"
        ]
        if c in results.columns
    ]

    st.dataframe(
        results[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # Predicted class distribution
    st.subheader("Transaction classification")
    class_counts = results["Prediction"].value_counts()
    st.bar_chart(class_counts)

    # Fraud probability distribution
    st.subheader("Fraud probability distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probabilities, bins=25)
    ax.axvline(
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold:.2f}"
    )
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Number of transactions")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    # Optional evaluation if true labels are provided
    if actual_labels is not None:
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix
        )

        y_true = pd.to_numeric(
            actual_labels, errors="coerce"
        )

        if y_true.isna().any() or not set(y_true.unique()).issubset({0, 1}):
            st.warning("Class must contain only 0 and 1 for evaluation.")
        else:
            y_true = y_true.astype(int)

            st.subheader("Evaluation against supplied labels")
            a, b, c = st.columns(3)
            a.metric(
                "Fraud precision",
                f"{precision_score(y_true, predictions, zero_division=0):.2%}"
            )
            b.metric(
                "Fraud recall",
                f"{recall_score(y_true, predictions, zero_division=0):.2%}"
            )
            c.metric(
                "Fraud F1-score",
                f"{f1_score(y_true, predictions, zero_division=0):.4f}"
            )

            cm = confusion_matrix(
                y_true, predictions, labels=[0, 1]
            )
            st.write(
                "Confusion matrix (rows = actual; columns = predicted)"
            )
            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=["Actual legitimate", "Actual fraud"],
                    columns=["Predicted legitimate", "Predicted fraud"]
                )
            )

    st.download_button(
        "Download prediction results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Upload a CSV to begin. A single-row CSV analyzes one transaction; "
        "a multi-row CSV analyzes a batch."
    )

st.divider()
st.caption(
    "For educational use only. A fraud prediction is an alert for review, "
    "not proof that a transaction is fraudulent."
)
