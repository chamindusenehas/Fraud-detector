Fraud Detection Dashboard
A Streamlit-based dashboard for detecting fraudulent transactions using a trained XGBoost model. The dashboard allows users to predict whether a transaction is fraudulent and visualize model performance and feature importance.

Project Overview
This project uses a simulated dataset of financial transactions to train an XGBoost model for fraud detection. The dataset includes Highly imbalanced data. (99.3% < Non-Fraud)

The model uses engineered features to get the model to an accurate and usable level.

Model Overview:         Displays precision, recall, F1 score, PR-AUC, and ROC-AUC, with an interactive feature importance chart.
Prediction Interface:   Allows users to input transaction details and predict fraud probability, with historical features approximated from precomputed lookups.



To use this,

Clone the Repository:
git clone https://github.com/chamindusenehas/Fraud-detector.git



Install Dependencies:
pip install -r requirements.txt



Run the Streamlit App:
streamlit run app.py



License
This project is licensed under the MIT License.

Last Updated: August 11, 2025