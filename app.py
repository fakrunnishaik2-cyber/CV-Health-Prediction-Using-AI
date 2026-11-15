
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Cardiovascular Health Analytics",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------
# CUSTOM DESIGN
# --------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071a35, #123b62);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div[data-testid="stMetric"] {
    background: #173e60;
    padding: 18px;
    border-radius: 12px;
}
div.stButton > button {
    background: #e63946;
    color: white;
    border-radius: 10px;
    height: 48px;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# LOAD DATASET
# --------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/CVD.csv")
    df = df.drop_duplicates()

    if "age" in df.columns:
        if df["age"].median() > 100:
            df["age_years"] = (df["age"] / 365.25).round(1)
        else:
            df["age_years"] = df["age"]

    df = df.drop(columns=["id", "age"], errors="ignore")

    return df

data = load_data()

# --------------------------------
# MODEL TRAINING
# --------------------------------

FEATURES = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active"
]

@st.cache_resource
def train_models(df):
    X = df[FEATURES]
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "SVM": LinearSVC(max_iter=5000, random_state=42)
    }

    scores = {}

    for name, model in models.items():
        if name == "Random Forest":
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)

        scores[name] = accuracy_score(y_test, predictions) * 100

    return models, scaler, scores

models, scaler, scores = train_models(data)

best_model = models["Logistic Regression"]

# Save model files
joblib.dump(best_model, "cvd_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# --------------------------------
# DASHBOARD HEADER
# --------------------------------

st.title("❤️ Cardiovascular Health Analytics Dashboard")

st.write(
    "Dataset Analysis, Statistics and Machine Learning Prediction"
)

st.divider()

# --------------------------------
# DATASET OVERVIEW
# --------------------------------

st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Records", f"{len(data):,}")

with c2:
    st.metric("Total Columns", len(data.columns))

with c3:
    st.metric("Missing Values", int(data.isnull().sum().sum()))

with c4:
    st.metric("Duplicate Records", int(data.duplicated().sum()))

st.divider()

# --------------------------------
# 1. CARDIOVASCULAR DISTRIBUTION
# --------------------------------

st.subheader("❤️ Cardiovascular Disease Distribution")

col1, col2 = st.columns(2)

with col1:
    cardio_counts = data["cardio"].value_counts().sort_index()

    st.bar_chart(
        cardio_counts,
        x_label="Cardiovascular Class",
        y_label="Number of Patients"
    )

with col2:
    st.write("Disease Distribution")

    st.write("Class 0:", int(cardio_counts.get(0, 0)))
    st.write("Class 1:", int(cardio_counts.get(1, 0)))

    st.write(
        "Class 0 represents the negative dataset class; "
        "Class 1 represents the positive dataset class."
    )

st.divider()

# --------------------------------
# 2. AGE DISTRIBUTION
# --------------------------------

st.subheader("👥 Age Distribution")

fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(data["age_years"], bins=25, edgecolor="black")
ax.set_xlabel("Age (Years)")
ax.set_ylabel("Number of Patients")
ax.set_title("Patient Age Distribution")

st.pyplot(fig)
plt.close(fig)

st.divider()

# --------------------------------
# 3. GENDER DISTRIBUTION
# --------------------------------

st.subheader("⚥ Gender Distribution")

gender_counts = data["gender"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 4))

ax.pie(
    gender_counts.values,
    labels=[f"Gender {x}" for x in gender_counts.index],
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title("Gender Distribution")

st.pyplot(fig)
plt.close(fig)

st.divider()

# --------------------------------
# 4. CHOLESTEROL ANALYSIS
# --------------------------------

st.subheader("🩸 Cholesterol Analysis")

cholesterol_counts = (
    data["cholesterol"].value_counts().sort_index()
)

st.bar_chart(
    cholesterol_counts,
    x_label="Cholesterol Category",
    y_label="Number of Patients"
)

st.divider()

# --------------------------------
# 5. BLOOD PRESSURE ANALYSIS
# --------------------------------

st.subheader("💓 Blood Pressure Analysis")

fig, ax = plt.subplots(figsize=(9, 4))

sample = data.sample(
    min(3000, len(data)), random_state=42
)

ax.scatter(
    sample["ap_hi"],
    sample["ap_lo"],
    alpha=0.35
)

ax.set_xlabel("Systolic Blood Pressure")
ax.set_ylabel("Diastolic Blood Pressure")
ax.set_title("Blood Pressure Distribution")

st.pyplot(fig)
plt.close(fig)

st.divider()

# --------------------------------
# 6. LIFESTYLE ANALYSIS
# --------------------------------

st.subheader("🏃 Lifestyle Analysis")

lifestyle = data.groupby("cardio")[
    ["smoke", "alco", "active"]
].mean() * 100

lifestyle.index = [
    "Class 0", "Class 1"
] if len(lifestyle) == 2 else lifestyle.index

lifestyle.columns = [
    "Smoking (%)",
    "Alcohol (%)",
    "Physical Activity (%)"
]

st.bar_chart(
    lifestyle,
    y_label="Percentage"
)

st.divider()

# --------------------------------
# 7. CORRELATION HEATMAP
# --------------------------------

st.subheader("🔥 Feature Correlation Heatmap")

corr = data.select_dtypes(
    include="number"
).corr()

fig, ax = plt.subplots(figsize=(11, 7))

im = ax.imshow(
    corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))

ax.set_xticklabels(
    corr.columns, rotation=90
)

ax.set_yticklabels(corr.columns)

ax.set_title("Feature Correlation")

fig.colorbar(im, ax=ax)

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.divider()

# --------------------------------
# 8. MODEL ACCURACY
# --------------------------------

st.subheader("🤖 Machine Learning Model Performance")

accuracy_df = pd.DataFrame({
    "Model": list(scores.keys()),
    "Accuracy (%)": list(scores.values())
})

st.bar_chart(
    accuracy_df.set_index("Model")
)

for name, score in scores.items():
    st.write(f"**{name}:** {score:.2f}%")

st.divider()

# --------------------------------
# 9. HEALTH PREDICTION
# --------------------------------

st.title("🫀 Patient Health Prediction")

st.write(
    "Enter patient information to get a machine-learning "
    "model prediction."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", 18, 100, 30)
    gender = st.selectbox("Gender", [1, 2])
    height = st.number_input("Height (cm)", 100, 220, 165)
    weight = st.number_input("Weight (kg)", 30, 200, 65)
    ap_hi = st.number_input(
        "Systolic Blood Pressure", 60, 250, 120
    )
    ap_lo = st.number_input(
        "Diastolic Blood Pressure", 40, 200, 80
    )

with col2:
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
    gluc = st.selectbox("Glucose", [1, 2, 3])
    smoke = st.selectbox("Smoking", [0, 1])
    alco = st.selectbox("Alcohol", [0, 1])
    active = st.selectbox("Physical Activity", [0, 1])

if st.button("❤️ PREDICT HEALTH"):

    input_data = pd.DataFrame(
        [[age, gender, height, weight,
          ap_hi, ap_lo, cholesterol, gluc,
          smoke, alco, active]],
        columns=FEATURES
    )

    scaled_data = scaler.transform(input_data)

    prediction = best_model.predict(scaled_data)

    if prediction[0] == 1:
        st.warning(
            "Model prediction: Positive class (1)"
        )
    else:
        st.success(
            "Model prediction: Negative class (0)"
        )

    st.caption(
        "Academic project demonstration only. "
        "This prediction is not a medical diagnosis."
    )

# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Cardiovascular Health Analytics | "
    "Python | Pandas | Scikit-learn | Streamlit"
)
