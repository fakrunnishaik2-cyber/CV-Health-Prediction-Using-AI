import pandas as pd

# Load the CVD dataset
data = pd.read_csv("data/CVD.csv")

# Display first five records
print("CV HEALTH PREDICTION DATASET")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())
# Display dataset shape
print("\nDataset Rows and Columns:")
print(data.shape)

print("\nChecking Missing Values:")
print(data.isnull().sum())

print("\nChecking Duplicate Records:")
print(data.duplicated().sum())

print("\nCardio Distribution:")
print(data["cardio"].value_counts())

# Remove duplicate records
data = data.drop_duplicates()

print("\nFinal Dataset Shape:")
print(data.shape)

print("\nData Preprocessing Completed!")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Separate input and output
X = data.drop(columns=["cardio", "age"])
y = data["cardio"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": LinearSVC(max_iter=5000, random_state=42)
}

# Train models
for name, model in models.items():

    if name == "Random Forest":
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel:", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nAll Models Trained Successfully!")

# STEP 5: SAVE BEST MODEL

import joblib

best_model = models["Logistic Regression"]

joblib.dump(best_model, "cvd_model.pkl")

joblib.dump(scaler, "scaler.pkl")

print("Best Model Saved Successfully!")


# STEP 6: STREAMLIT WEBSITE

# STEP 6: PROFESSIONAL STREAMLIT WEBSITE

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="CV Health Prediction",
    page_icon="❤️",
    layout="wide"
)

# Load saved model and scaler
best_model = joblib.load("cvd_model.pkl")
scaler = joblib.load("scaler.pkl")

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071a35, #123b62);
    color: white;
}
.main-title {
    text-align: center;
    color: #ffffff;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #ff758f;
    font-size: 18px;
}
.heart {
    text-align: center;
    font-size: 110px;
    animation: beat 1s infinite;
}
@keyframes beat {
    0%, 100% {transform: scale(1);}
    50% {transform: scale(1.12);}
}
.info-card {
    background: #ffffff;
    color: #12304a;
    padding: 22px;
    border-radius: 16px;
    text-align: center;
}
div.stButton > button {
    background: #e63946;
    color: white;
    border-radius: 12px;
    border: none;
    height: 48px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">CARDIOVASCULAR HEALTH PREDICTION</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Artificial Intelligence & Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="heart">🫀</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
<h3>Welcome to Heart Health Prediction</h3>
<p>Enter the details below to see the machine-learning model prediction.</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 18, 100, 30)
    gender = st.selectbox("Gender", [1, 2])
    height = st.number_input("Height (cm)", 100, 220, 165)
    weight = st.number_input("Weight (kg)", 30, 200, 65)
    ap_hi = st.number_input("Systolic Blood Pressure", 60, 250, 120)
    ap_lo = st.number_input("Diastolic Blood Pressure", 40, 200, 80)

with col2:
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
    gluc = st.selectbox("Glucose", [1, 2, 3])
    smoke = st.selectbox("Smoking", [0, 1])
    alco = st.selectbox("Alcohol", [0, 1])
    active = st.selectbox("Physical Activity", [0, 1])

st.write("")

if st.button("❤️ PREDICT HEALTH"):
    input_data = pd.DataFrame(
        [[age, gender, height, weight, ap_hi, ap_lo,
          cholesterol, gluc, smoke, alco, active]],
        columns=[
            "age_years", "gender", "height", "weight",
            "ap_hi", "ap_lo", "cholesterol", "gluc",
            "smoke", "alco", "active"
        ]
    )

    input_data = input_data.reindex(
        columns=scaler.feature_names_in_
    )

    scaled_data = scaler.transform(input_data)
    prediction = best_model.predict(scaled_data)

    if prediction[0] == 1:
        st.warning("Model prediction: Positive class (1)")
    else:
        st.success("Model prediction: Negative class (0)")




