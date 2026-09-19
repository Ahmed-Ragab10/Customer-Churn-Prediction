import os
import joblib
import pandas as pd
import streamlit as st

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction", page_icon="📊", layout="wide"
)

# =========================
# Load Model
# =========================
MODEL_PATH = "models/churn_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# =========================
# Title
# =========================
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information to predict whether the customer will churn."
)
st.divider()

# =========================
# Inputs in 3 Columns
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Customer Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=100, value=12
    )

with col2:
    st.header("🌐 Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["Yes", "No", "No phone service"]
    )
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )

with col3:
    st.header("💳 Contract & Charges")
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=80.5, step=0.1
    )
    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=966.0, step=0.1
    )

st.divider()

# =========================
# Prediction Button
# =========================
if st.button(" Predict Churn", use_container_width=True):
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    customer_df = pd.DataFrame([customer])

    prediction = model.predict(customer_df)[0]
    probabilities = model.predict_proba(customer_df)[0]
    classes = list(model.classes_)
    churn_probability = probabilities[classes.index("Yes")]

    st.subheader("Prediction Result")
    if prediction == "Yes":
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    st.metric("Churn Probability", f"{churn_probability:.2%}")
    st.progress(float(churn_probability))