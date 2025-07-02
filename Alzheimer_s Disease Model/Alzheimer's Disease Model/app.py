# app.py  –  Streamlit Alzheimer’s prediction dashboard
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ #
# Streamlit page setup
st.set_page_config(page_title="Alzheimer's Prediction App", layout="wide")
st.title("🧠 Alzheimer's Disease Prediction Dashboard")
st.write("Upload a patient dataset, train a model, explore graphs, "
         "and predict Alzheimer's diagnosis.")

LABEL_MAP = {0: "No Alzheimer's", 1: "Alzheimer's"}

# ------------------------------------------------------------------ #
# 1. Load data
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df

uploaded_file = st.file_uploader("📄 Upload patient dataset (.csv)", type="csv")

if uploaded_file is None:
    st.info("⬆️ Upload a CSV file to get started.")
    st.stop()

df = load_data(uploaded_file)
st.subheader("🔍 Data preview")
st.dataframe(df.head())

if "Diagnosis" not in df.columns:
    st.error("❌ Dataset must contain a **Diagnosis** column (0 = No AD, 1 = AD).")
    st.stop()

# ------------------------------------------------------------------ #
# 2. Model selection
st.subheader("⚙️ Select a Machine-Learning Model")
model_option = st.selectbox(
    "Choose classifier",
    ["Random Forest", "Logistic Regression", "Support Vector Machine"]
)

# ------------------------------------------------------------------ #
# 3. Train model
@st.cache_resource
def train_model(data: pd.DataFrame, model_choice: str):
    X = data.drop(columns=["PatientID", "DoctorInCharge", "Diagnosis"], errors="ignore")
    y = data["Diagnosis"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_choice == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=2000, random_state=42)
    else:
        model = SVC(probability=True, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model, report, acc, X_test, y_test, y_pred, scaler, X.columns

model, report, acc, X_test, y_test, y_pred, scaler, feature_names = train_model(
    df, model_option
)

st.success(f"✅ Model trained – accuracy {acc*100:.2f}%")

# ------------------------------------------------------------------ #
# 4. Evaluation metrics and plots
st.subheader("📋 Classification Report (Precision • Recall • F1-Score)")
report_df = (
    pd.DataFrame(report).transpose()
    .rename(index=LABEL_MAP, errors="ignore")
    .round(3)
)
report_display = report_df.drop(columns=["support"], errors="ignore")
st.dataframe(report_display, use_container_width=True)

# Metrics at a glance
col1, col2, col3 = st.columns(3)
col1.metric("Overall Accuracy", f"{acc:.3f}")
if "1" in report:
    col2.metric("F1 (AD)", f"{report['1']['f1-score']:.3f}")
    col3.metric("Recall (AD)", f"{report['1']['recall']:.3f}")

# Side-by-side plots: Confusion Matrix and ROC Curve
plot1, plot2 = st.columns(2)

with plot1:
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[LABEL_MAP[0], LABEL_MAP[1]],
        yticklabels=[LABEL_MAP[0], LABEL_MAP[1]],
        ax=ax_cm
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

with plot2:
    st.subheader("📈 ROC Curve")
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], "k--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)

        st.info(f"**Area Under Curve (AUC):** {roc_auc:.3f}")
    else:
        st.info("ROC curve unavailable (model has no predict_proba).")

# Feature importance (only for RF)
if model_option == "Random Forest":
    st.subheader("📌 Feature Importance")
    importance_df = (
        pd.DataFrame({"Feature": feature_names,
                      "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
    )
    fig_imp, ax_imp = plt.subplots(figsize=(6, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax_imp)
    st.pyplot(fig_imp)

# ------------------------------------------------------------------ #
# 5. Variable distribution explorer
st.subheader("📊 Explore a Variable")
choice_var = st.selectbox(
    "Select variable", df.drop(columns=["PatientID", "DoctorInCharge"]).columns
)

fig_var, ax_var = plt.subplots(figsize=(6, 4))
if pd.api.types.is_numeric_dtype(df[choice_var]):
    sns.histplot(df[choice_var], kde=True, bins=30, ax=ax_var)
else:
    sns.countplot(x=df[choice_var], ax=ax_var)
st.pyplot(fig_var)

# ------------------------------------------------------------------ #
# 6. Manual prediction
st.subheader("🔮 Predict for a New Patient")
input_data = {}
for col in feature_names:
    if np.issubdtype(df[col].dtype, np.number):
        input_data[col] = st.number_input(
            col,
            value=float(df[col].mean()),
            help=f"Range in data: {df[col].min()} – {df[col].max()}"
        )

if st.button("Predict diagnosis"):
    input_scaled = scaler.transform(pd.DataFrame([input_data]))
    pred = model.predict(input_scaled)[0]
    conf = model.predict_proba(input_scaled)[0][pred] if hasattr(model, "predict_proba") else 1.0
    label = LABEL_MAP[pred]
    emoji = "🧠" if pred == 1 else "✅"
    st.markdown(f"### {emoji} Prediction: **{label}**")
    st.write(f"Confidence: **{conf*100:.2f}%**")
