import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Diabetes Class Prediction", layout="wide")

# -----------------------------
# Corporate palette + UI CSS
# -----------------------------
CSS = """
<style>
:root{
  --primary:#1E3A8A;      /* blue-800 */
  --primary-600:#2563EB;  /* blue-600 */
  --bg:#F8FAFC;           /* slate-50 */
  --card:#FFFFFF;
  --border:#E5E7EB;
  --text:#0F172A;         /* slate-900 */
  --muted:#64748B;        /* slate-500 */
}

html, body, [class*="css"]{
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Ubuntu, "Helvetica Neue", Arial, sans-serif;
  color: var(--text);
  background: var(--bg);
}

/* Headline */
h1, h2, h3 { color: var(--text); }

/* Cards */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
  box-shadow: 0 1px 2px rgba(16,24,40,.06);
  margin-bottom: 16px;
}
.card-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 8px; }

/* Buttons */
.stButton>button {
  background: var(--primary);
  color: #fff;
  border: 1px solid var(--primary);
  padding: .5rem 1rem;
  border-radius: .5rem;
}
.stButton>button:hover { background: var(--primary-600); border-color: var(--primary-600); }

/* Badges */
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background:#EEF2FF; color:var(--primary); border:1px solid #E0E7FF; font-weight:600; font-size:.85rem;
}

/* Divider & meta text */
.hr { height:1px; background:var(--border); border:0; margin:16px 0; }
.caption, .footer, .disclaimer { color: var(--muted); }

/* Disclaimer */
.disclaimer {
  font-size:.85rem; border-top:1px solid var(--border);
  padding-top:10px; margin-top:8px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model(path: str = "rf_model.pkl"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p.resolve()}")
    return joblib.load(p)

try:
    model = load_model("rf_model.pkl")
except Exception as e:
    st.error("Model could not be loaded. Ensure 'rf_model.pkl' is present next to this app.")
    st.exception(e)
    st.stop()

CLASS_LABELS = {0: "Non-Diabetic", 1: "Diabetic", 2: "Pre-Diabetic"}

# -----------------------------
# Header
# -----------------------------
st.title("Diabetes Class Prediction")
st.markdown("Estimate the **diabetes classification** using a trained Random Forest model.")

# -----------------------------
# Two-column layout (inputs | results)
# -----------------------------
col_left, col_right = st.columns([1, 1.1], gap="large")

# ---- Left: Input form
with col_left:
    with st.form("diab_form", clear_on_submit=False):
        st.markdown('<div class="card"><div class="card-title">Patient Information</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            gender = st.radio("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            hba1c = st.number_input("HbA1c (%)", min_value=1.0, max_value=15.0, value=6.0, step=0.1)
        with c2:
            chol = st.number_input("Cholesterol (mmol/L)", min_value=0.1, max_value=15.0, value=5.0, step=0.1)
            urea = st.number_input("Urea (mmol/L)", min_value=0.1, max_value=40.0, value=5.0, step=0.1)
            cr   = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            tg   = st.number_input("Triglycerides (mmol/L)", min_value=0.1, max_value=15.0, value=2.0, step=0.1)
        c3, c4 = st.columns(2)
        with c3:
            hdl  = st.number_input("HDL (mmol/L)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        with c4:
            ldl  = st.number_input("LDL (mmol/L)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
            vldl = st.number_input("VLDL (mg/dL)", min_value=0.1, max_value=80.0, value=5.0, step=0.1)

        submitted = st.form_submit_button("Run Prediction")
        st.markdown('</div>', unsafe_allow_html=True)

# ---- Right: Results panel
with col_right:
    st.markdown('<div class="card"><div class="card-title">Prediction Summary</div>', unsafe_allow_html=True)

    if submitted:
        invalid = any(v <= 0 for v in [age, bmi, hba1c, urea, cr])
        if invalid:
            st.error("Some fields must be strictly positive (Age, BMI, HbA1c, Urea, Creatinine).")
        else:
            X = pd.DataFrame([[
                gender, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi
            ]], columns=["Gender", "AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"])

            y = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]

            st.markdown(f"**Predicted class:** <span class='badge'>{CLASS_LABELS[y]}</span>", unsafe_allow_html=True)

            notes = {
                0: "No immediate sign of diabetes. Maintain a healthy lifestyle.",
                1: "High likelihood of diabetes. Medical follow-up recommended.",
                2: "Risk of pre-diabetes. Monitor closely and adopt preventive habits."
            }
            st.caption(notes[y])

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            st.markdown("**Class probabilities**")

            dfp = pd.DataFrame({
                "Class": [CLASS_LABELS[i] for i in range(len(proba))],
                "Probability": proba
            }).sort_values("Probability", ascending=False, ignore_index=True)
            dfp["Probability"] = dfp["Probability"].map(lambda p: f"{p:.2%}")
            st.dataframe(dfp, use_container_width=True, hide_index=True)
    else:
        st.caption("Fill the form and click **Run Prediction** to see results here.")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Disclaimer & Footer
# -----------------------------
st.markdown(
    '<div class="disclaimer">'
    'This tool is intended for educational and research purposes only. '
    'It must not be used as a substitute for professional medical advice, diagnosis, or treatment.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('<hr class="hr">', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">'
    'Author: <strong>Nasser Chaouchi</strong> &nbsp;|&nbsp; '
    '<a href="https://www.linkedin.com/in/nasser-chaouchi/" target="_blank">LinkedIn</a> &nbsp;|&nbsp; '
    '<a href="https://github.com/nasser-chaouchi" target="_blank">GitHub</a>'
    '</div>',
    unsafe_allow_html=True
)
