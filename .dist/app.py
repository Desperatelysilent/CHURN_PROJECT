import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction Tool")
st.write("Enter customer details below to predict their probability of canceling their service.")

# 1. Load trained model and expected feature names
@st.cache_resource
def load_assets():
    model = joblib.load('churn_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

try:
    model, model_columns = load_assets()
except Exception as e:
    st.error("Error loading model files! Make sure 'churn_model.pkl' and 'model_columns.pkl' exist in your directory.")
    st.stop()

# 2. Collect User Inputs
tenure = st.slider("Tenure (Months with company)", min_value=1, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

if st.button("Predict Churn Risk"):
    # 3. Create a DataFrame initialized with 0s matching exact feature names and order
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # 4. Map numerical fields
    if 'tenure' in input_df.columns:
        input_df['tenure'] = tenure
    if 'MonthlyCharges' in input_df.columns:
        input_df['MonthlyCharges'] = monthly_charges
    if 'TotalCharges' in input_df.columns:
        input_df['TotalCharges'] = tenure * monthly_charges

    # 5. Map categorical features (One-Hot Encoded columns)
    raw_inputs = [
        f"Contract_{contract}",
        f"InternetService_{internet_service}",
        f"PaymentMethod_{payment_method}"
    ]
    
    for col in raw_inputs:
        if col in input_df.columns:
            input_df[col] = 1

    # 6. Generate Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk!** Probability of canceling: **{probability:.1%}**")
    else:
        st.success(f"✅ **Low Churn Risk.** Probability of canceling: **{probability:.1%}**")