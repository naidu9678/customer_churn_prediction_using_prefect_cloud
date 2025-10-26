# =============================================================
# Customer Churn Prediction Workflow using Prefect Cloud
# =============================================================
# Business Understanding:
# Problem: Predict customer churn for a telecom company.
# Goal: Identify customers likely to leave, enabling better retention strategies.
# =============================================================

import pandas as pd
import numpy as np
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------------------------------------
# Step 1: Load Dataset
# ----------------------------------------------------------
@task
def load_dataset():
    csv_path = Path(__file__).parent.parent / "data" / "Telco-Customer-Churn.csv"
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully from:", csv_path)
    print(f"Shape: {df.shape}")
    return df


# ----------------------------------------------------------
# Step 2: Data Preprocessing
# ----------------------------------------------------------
@task(log_prints=True)
def preprocess_data(df):
    print("\nStarting data preprocessing...")

    print("\nData Types:")
    print(df.dtypes)

    print("\nSummary Statistics:")
    print(df.describe(include='all').T)

    print("\nMissing Values Before Imputation:")
    print(df.isna().sum())

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print("\nMissing Values After Imputation:")
    print(df.isna().sum())

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                                    labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])
        le = LabelEncoder()
        df['tenure_group'] = le.fit_transform(df['tenure_group'])

    scaler = MinMaxScaler()
    features = df.drop('Churn', axis=1)
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    scaled_features['Churn'] = df['Churn']

    print("\nData preprocessing completed successfully")

    # Artifact: Data Preprocessing Summary
    preprocessing_summary = (
        "### Data Preprocessing Summary\n"
        f"- Records: {scaled_features.shape[0]}\n"
        f"- Features: {scaled_features.shape[1]}\n"
        "- Missing values handled and categorical columns encoded\n"
        "- Numerical features normalized using MinMaxScaler"
    )
    create_markdown_artifact(key="preprocessing-summary", markdown=preprocessing_summary)

    return scaled_features


# ----------------------------------------------------------
# Step 3: Exploratory Data Analysis (EDA)
# ----------------------------------------------------------
@task(log_prints=True)
def perform_eda(df):
    print("\nPerforming Exploratory Data Analysis...")

    os.makedirs("eda_outputs", exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.savefig("eda_outputs/churn_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm")
    plt.title('Feature Correlation Heatmap')
    plt.savefig("eda_outputs/correlation_heatmap.png")
    plt.close()

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    importances[:10].plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.savefig("eda_outputs/feature_importance.png")
    plt.close()

    print("EDA completed. Charts saved in 'eda_outputs/' folder.")
    print("\nTop correlated features with Churn:")
    print(df.corr()['Churn'].abs().sort_values(ascending=False).head(6))

    # Artifact: EDA Summary
    eda_summary = (
        "### EDA Summary\n"
        "- Generated churn distribution, correlation heatmap, and feature importance plots\n"
        "- Charts saved locally in `eda_outputs/`\n"
        "- Top correlated features with Churn displayed in Prefect logs"
    )
    create_markdown_artifact(key="eda-summary", markdown=eda_summary)

    return importances


# ----------------------------------------------------------
# Step 4: Model Training and Evaluation
# ----------------------------------------------------------
@task(log_prints=True)
def train_and_evaluate(df):
    print("\nStarting model training and evaluation...")

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
            "F1 Score": round(f1_score(y_test, y_pred), 3)
        }

        results[name] = metrics

    print("\nModel Evaluation Metrics:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    # Artifact: Model Evaluation Report
    markdown_report = "### Model Evaluation Results\n"
    for model, scores in results.items():
        markdown_report += f"\n**{model}**\n"
        for metric, value in scores.items():
            markdown_report += f"- {metric}: `{value}`\n"

    create_markdown_artifact(key="model-evaluation-report", markdown=markdown_report)

    return results


# ----------------------------------------------------------
# Step 5: Prefect Flow Definition
# ----------------------------------------------------------
@flow(log_prints=True)
def churn_prediction_workflow():
    print("\nStarting Customer Churn Prediction Workflow...")
    df = load_dataset()
    processed_df = preprocess_data(df)
    perform_eda(processed_df)
    metrics = train_and_evaluate(processed_df)

    # Artifact: Workflow Summary
    summary = (
        "### Workflow Summary\n"
        "- Loaded dataset and preprocessed data\n"
        "- Completed EDA with saved visual reports\n"
        "- Trained and evaluated models\n"
        "- Artifacts stored in Prefect Cloud"
    )
    create_markdown_artifact(key="workflow-summary", markdown=summary)

    print("\nWorkflow execution completed successfully.")
    return metrics


# ----------------------------------------------------------
# Step 6: Serve the Workflow (Run every 2 minutes)
# ----------------------------------------------------------
if __name__ == "__main__":
    churn_prediction_workflow.serve(
        name="customer-churn-pipeline",
        tags=["data_pipeline", "mlops", "churn"],
        parameters={},
        interval=120  # 2 minutes
    )
