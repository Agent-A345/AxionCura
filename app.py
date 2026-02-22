import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AxionCura",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* ================= BACKGROUND ================= */
.stApp {
    background: linear-gradient(135deg, #4F7C82, #0B2E33);
    color: #ffffff;
}

/* ================= TITLES ================= */
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    opacity: 0.9;
    margin-bottom: 30px;
}

/* ================= SECTION ================= */
.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 20px;
    color: #ffffff;
}

/* ================= DISCLAIMER ================= */
.disclaimer {
    background: rgba(255, 255, 255, 0.18);
    padding: 18px;
    border-left: 6px solid #4F7C82;
    border-radius: 10px;
    font-size: 14px;
    margin-bottom: 25px;
}

/* ================= BUTTON ================= */
.stButton > button {
    background: #0B2E33;
    color: white;
    border-radius: 14px;
    padding: 12px 22px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: #4F7C82;
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 12px 28px rgba(0,0,0,0.35);
}

/* ================= RESULT BOXES ================= */
.result-high {
    background: rgba(220, 53, 69, 0.18);
    padding: 20px;
    border-left: 6px solid #dc3545;
    border-radius: 12px;
}
.result-low {
    background: rgba(40, 167, 69, 0.18);
    padding: 20px;
    border-left: 6px solid #28a745;
    border-radius: 12px;
}

/* ================= LABEL VISIBILITY ================= */
label,
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
div[data-baseweb="typography"] {
    color: #ffffff !important;
}

/* ================= SELECT PLACEHOLDER ================= */
div[data-baseweb="select"] span {
    color: #0B2E33 !important;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("stroke_pipeline.joblib")

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🧠 AxionCura </div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Early Stroke Risk Detection using Machine Learning</div>", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class="disclaimer">
⚠️ <b>IMPORTANT MEDICAL DISCLAIMER</b><br>
This AI tool is for educational purposes only and should NOT replace professional medical advice.
</div>
""", unsafe_allow_html=True)

# ---------------- PATIENT INFO ----------------
st.markdown("<div class='section-title'>🩺 Patient Information</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    gender = st.selectbox("Gender", ["Select", "Male", "Female"], index=0)
    age = st.number_input("Age", 1, 100, 45)
    hypertension = st.selectbox("Hypertension", ["Select", 0, 1], index=0)
    heart_disease = st.selectbox("Heart Disease", ["Select", 0, 1], index=0)
    ever_married = st.selectbox("Ever Married", ["Select", "Yes", "No"], index=0)

with c2:
    work_type = st.selectbox("Work Type", ["Select", "Private", "Self-employed", "Govt_job", "children"], index=0)
    residence_type = st.selectbox("Residence Type", ["Select", "Urban", "Rural"], index=0)
    glucose = st.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 110.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 26.0)
    smoking = st.selectbox("Smoking Status", ["Select", "never smoked", "formerly smoked", "smokes"], index=0)

analyze = st.button("🔍 Analyze Stroke Risk")

# ---------------- VALIDATION ----------------
if analyze:
    if "Select" in [
        gender, hypertension, heart_disease, ever_married,
        work_type, residence_type, smoking
    ]:
        st.warning("⚠️ Please select all dropdown fields before analysis.")
    else:
        input_df = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "residence_type": residence_type,
            "avg_glucose_level": glucose,
            "bmi": bmi,
            "smoking_status": smoking
        }])

        pred = model.predict(input_df)[0]

        if pred == 1:
            st.markdown("""
            <div class="result-high">
            🚨 <b>High Stroke Risk Detected</b><br>
            Please consult a medical professional immediately.
            </div>
            """, unsafe_allow_html=True)
            st.progress(85)
        else:
            st.markdown("""
            <div class="result-low">
            ✅ <b>Low Stroke Risk Detected</b><br>
            Maintain a healthy lifestyle and regular checkups.
            </div>
            """, unsafe_allow_html=True)
            st.progress(25)


