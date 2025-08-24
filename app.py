import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

terminal_fraud_rate = joblib.load('terminal_fraud_rate.pkl')
customer_fraud_rate = joblib.load('customer_fraud_rate.pkl')
high_risk_hours = joblib.load('high_risk_hours.pkl')
customer_tx_count_dict = joblib.load('customer_tx_count_dict.pkl')
terminal_tx_count_dict = joblib.load('terminal_tx_count_dict.pkl')
avg_customer_tx_amount_dict = joblib.load('avg_customer_tx_amount_dict.pkl')
customer_std_dict = joblib.load('customer_std_dict.pkl')
median_terminal_amount_dict = joblib.load('median_terminal_amount_dict.pkl')
customer_avg_time_dict = joblib.load('customer_avg_time_dict.pkl')


precision = 0.7938
recall = 0.5613
f1 = 0.6576
pr_auc = 0.7621 
roc_auc = 0.9950

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon=":shield:", layout="wide")


st.title("Fraud Detection Dashboard")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", [ "Predict Fraud","Model Overview & Feature Importance"])

if page == "Model Overview & Feature Importance":
    st.header("Model Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precision", f"{precision:.4f}")
    col2.metric("Recall", f"{recall:.4f}")
    col3.metric("F1 Score", f"{f1:.4f}")
    col4.metric("PR-AUC", f"{pr_auc:.4f}")
    col5.metric("ROC-AUC", f"{roc_auc:.4f}")

    st.header("Feature Importance Analysis")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)


    st.table(importance_df)


    st.bar_chart(importance_df.set_index('Feature')['Importance'])

elif page == "Predict Fraud":
    st.header("Predict Fraud for a New Transaction")
    st.write("Input the transaction details below. Some features require historical context (e.g., time since last transaction), which you need to provide based on the customer's history. For unknown customers/terminals, defaults (e.g., 0) will be used.")


    with st.form(key="prediction_form"):

        col1, col2 = st.columns(2)
        with col1:
            tx_datetime_str = st.text_input("Transaction Datetime (YYYY-MM-DD HH:MM:SS)", value="2018-04-01 00:00:31", help="Enter a valid datetime format.")
            customer_id = st.number_input("Customer ID", value=596, step=1, help="Unique customer identifier.")
            terminal_id = st.number_input("Terminal ID", value=3156, step=1, help="Unique terminal/merchant identifier.")
        with col2:
            tx_amount = st.number_input("Transaction Amount", value=57.16, step=0.01, min_value=0.0, help="Must be non-negative.")
            tx_time_days = st.number_input("Transaction Time Days (days since start)", value=0, step=1, min_value=0, help="Days elapsed since reference start date.")


        with st.expander("History-Dependent Features"):
            time_since_last_tx = st.number_input("Time Since Last Transaction (seconds)", value=0.0, step=1.0, min_value=0.0, help="Seconds since customer's last transaction.")
            amount_change = st.number_input("Amount Change from Previous Transaction", value=0.0, step=0.01, help="Difference from previous amount (can be negative).")
            tx_count_10min = st.number_input("Transaction Count in Last 10 Minutes", value=1, step=1, min_value=1, help="Number of transactions in the last 10 minutes.")

        submit_button = st.form_submit_button(label="Predict")

    if submit_button:

        if tx_amount < 0:
            st.error("Transaction Amount must be non-negative.")
        else:
            try:

                tx_datetime = datetime.strptime(tx_datetime_str, "%Y-%m-%d %H:%M:%S")
                hour = tx_datetime.hour
                day_of_week = tx_datetime.weekday()
                is_weekend = 1 if day_of_week in [5, 6] else 0
                log_tx_amount = np.log1p(tx_amount)
                is_high_risk_hour = 1 if hour in high_risk_hours else 0


                cust_tx_count = customer_tx_count_dict.get(customer_id, 0) + 1
                term_tx_count = terminal_tx_count_dict.get(terminal_id, 0) + 1
                avg_cust_tx_amt = avg_customer_tx_amount_dict.get(customer_id, 0)
                amount_deviation = tx_amount / (avg_cust_tx_amt + 1e-6)
                term_fraud_rate = terminal_fraud_rate.get(terminal_id, 0)
                cust_fraud_rate = customer_fraud_rate.get(customer_id, 0)
                cust_std = customer_std_dict.get(customer_id, 0)
                amount_outlier = 1 if abs(tx_amount - avg_cust_tx_amt) > 2 * cust_std else 0
                term_amt_dev_median = tx_amount / (median_terminal_amount_dict.get(terminal_id, 1) + 1e-6)
                cust_avg_time = customer_avg_time_dict.get(customer_id, 0)
                tx_interval_deviation = time_since_last_tx / (cust_avg_time + 1e-6)


                features = pd.DataFrame(
                    [[
                        tx_amount, tx_time_days, hour, day_of_week, is_weekend,
                        cust_tx_count, term_tx_count, log_tx_amount, time_since_last_tx,
                        avg_cust_tx_amt, amount_deviation, term_fraud_rate, cust_fraud_rate,
                        amount_outlier, tx_count_10min, amount_change, term_amt_dev_median,
                        tx_interval_deviation, is_high_risk_hour
                    ]],
                    columns=feature_names
                )


                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                prediction_proba = model.predict_proba(features_scaled)[0][1]


                st.success(f"Prediction: {'Fraud' if prediction == 1 else 'Not Fraud'}")
                st.metric("Fraud Probability", f"{prediction_proba:.4f}")
                st.progress(int(prediction_proba))
            except ValueError as e:
                st.error(f"Error: {e}. Please check input formats.")


st.markdown('<div class="footer">Dashboard Updated: August 24, 2025</div>', unsafe_allow_html=True)