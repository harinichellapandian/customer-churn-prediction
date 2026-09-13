import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
model = joblib.load("model.pkl")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Business Problem", "Data Insights", "Churn Prediction"]
)

# -------------------------------------------------
# PAGE 1: BUSINESS PROBLEM
# -------------------------------------------------
if page == "Business Problem":

    st.title("📊 Customer Churn Prediction")

    st.header("Business Problem")

    st.write(
        """
        Customer churn is an important business problem for telecommunications
        companies. Losing existing customers can reduce revenue and increase
        the cost of acquiring new customers.

        This project uses customer demographic, service usage, contract and
        billing information to predict whether a customer is likely to churn.
        """
    )

    st.header("Business Objective")

    st.write(
        """
        The objective is to develop a machine learning classification model
        that can identify customers who are likely to churn and support
        customer retention decisions.
        """
    )

    st.header("Dataset")

    st.write(
        """
        The dataset contains customer information including tenure,
        contract type, internet services, payment method, monthly charges,
        total charges and churn status.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", len(df))

    with col2:
        st.metric("Churned Customers", int((df["Churn"] == "Yes").sum()))

    with col3:
        st.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean():.2%}")


# -------------------------------------------------
# PAGE 2: DATA INSIGHTS
# -------------------------------------------------
elif page == "Data Insights":

    st.title("📈 Data Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", len(df))

    with col2:
        st.metric(
            "Customers Churned",
            int((df["Churn"] == "Yes").sum())
        )

    with col3:
        st.metric(
            "Average Monthly Charges",
            f"₹{df['MonthlyCharges'].mean():.2f}"
        )

    st.subheader("Customer Churn Distribution")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Churn", ax=ax)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.subheader("Contract Type and Customer Churn")

    fig, ax = plt.subplots()
    sns.countplot(
        data=df,
        x="Contract",
        hue="Churn",
        ax=ax
    )
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Number of Customers")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    st.subheader("Monthly Charges and Customer Churn")

    fig, ax = plt.subplots()
    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        ax=ax
    )
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges")
    st.pyplot(fig)


# -------------------------------------------------
# PAGE 3: CHURN PREDICTION
# -------------------------------------------------
else:

    st.title("🔮 Customer Churn Prediction")

    st.write(
        "Enter the customer's details below to predict churn probability."
    )

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=800.0
        )

    if st.button("🔍 Predict Churn"):

        input_data = pd.DataFrame([{
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
            "TotalCharges": total_charges
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:

            st.error("⚠️ Prediction: Customer is likely to CHURN")

            st.write(
                f"Estimated churn probability: {probability:.2%}"
            )

            st.warning(
                "Business Recommendation: Consider targeted customer "
                "retention measures for this customer."
            )

        else:

            st.success("✅ Prediction: Customer is likely to STAY")

            st.write(
                f"Estimated churn probability: {probability:.2%}"
            )

            st.info(
                "Business Recommendation: Continue providing good "
                "service and monitor customer satisfaction."
            )
