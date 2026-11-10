
import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature columns
model = joblib.load("customer_churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Customer Churn Prediction")
st.title("Customer Churn Prediction System")

st.write(
    "Enter customer information to predict whether "
    "the customer may leave the company."
)

# Customer information form
with st.form("customer_form"):

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )

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

    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )
    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=840.0
    )

    submit = st.form_submit_button("Predict Churn")


if submit:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Apply the same encoding used during training
   customer_encoded = pd.DataFrame(
    0, index=[0], columns=model_columns
)

numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for col in numeric_columns:
    customer_encoded[col] = customer[col].iloc[0]

for col in model_columns:
    if "_" in col:
        feature, category = col.split("_", 1)

        if feature in customer.columns:
            if str(customer.loc[0, feature]) == category:
                customer_encoded.loc[0, col] = 1

    prediction = model.predict(customer_encoded)[0]
    probability = model.predict_proba(customer_encoded)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("This customer may leave the company.")
    else:
        st.success("This customer is likely to stay.")

    st.write(f"Churn Probability: {probability * 100:.2f}%")
