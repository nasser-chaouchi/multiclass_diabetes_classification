import streamlit as st
import joblib
import pandas as pd

model = joblib.load("rf_model.pkl")

class_labels = {
    0: "Non-Diabetic",
    1: "Diabetic",
    2: "Pre-Diabetic"
}

st.title("🩺 Diabetes Class Prediction")
st.write("Enter patient data below to predict diabetes classification.")

gender = st.radio("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
age = st.number_input("Age", min_value=1, max_value=100, value=50)
bmi = st.number_input("BMI", min_value=1.0, max_value=60.0, value=25.0, step=0.1)
hba1c = st.number_input("HbA1c (%)", min_value=1.0, max_value=15.0, value=6.0, step=0.1)
chol = st.number_input("Cholesterol", min_value=0.1, max_value=10.0, value=5.0, step=0.1)
urea = st.number_input("Urea", min_value=0.1, max_value=30.0, value=5.0, step=0.1)
cr = st.number_input("Creatinine", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
tg = st.number_input("Triglycerides", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
hdl = st.number_input("HDL", min_value=0.1, max_value=4.0, value=1.0, step=0.1)
ldl = st.number_input("LDL", min_value=0.1, max_value=6.0, value=2.0, step=0.1)
vldl = st.number_input("VLDL", min_value=0.1, max_value=30.0, value=5.0, step=0.1)

X_input = pd.DataFrame([[
    gender, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi
]], columns=["Gender", "AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"])

if st.button("Predict"):

    if any([
        age <= 0,
        bmi <= 0,
        hba1c <= 0,
        urea <= 0,
        cr <= 0,
    ]):
        st.error("🚫 Some fields must be strictly positive (e.g. Age, BMI, HbA1c, Urea, Creatinine). Please check your input.")

    else:
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

        st.markdown(f"### 🧾 Predicted Class: **{class_labels[prediction]}**")

        explanations = {
            0: "✅ No immediate sign of diabetes. Maintain a healthy lifestyle.",
            1: "⚠️ High likelihood of diabetes. Medical follow-up recommended.",
            2: "⚠️ Risk of pre-diabetes. Monitor closely and adopt preventive habits."
        }
        st.info(explanations[prediction])

        proba_df = pd.DataFrame({
            "Class": [class_labels[i] for i in range(len(proba))],
            "Probability": [f"{p:.2%}" for p in proba]
        })

        st.markdown("### 🔢 Class Probabilities")
        st.table(proba_df)