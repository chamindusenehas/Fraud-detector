<h1>Fraud Detection Dashboard</h1>
A Streamlit-based dashboard for detecting fraudulent transactions using a trained XGBoost model. The dashboard allows users to predict whether a transaction is fraudulent and visualize model performance and feature importance.<br><br>

<h2>Project Overview</h2>
This project uses a simulated dataset of financial transactions to train an XGBoost model for fraud detection. The dataset includes Highly imbalanced data. (99.3% < Non-Fraud)

The model uses engineered features to get the model to an accurate and usable level.<br>

<h3><b>Model Overview:</b></h3>   Displays precision, recall, F1 score, PR-AUC, and ROC-AUC, with an interactive feature importance chart.
<h3><b>Prediction Interface:</b></h3> Allows users to input transaction details and predict fraud probability, with historical features approximated from precomputed lookups.<br>
<br>
<br>



<i><b>To use this,<br></b></i>

Clone the Repository:
git clone https://github.com/chamindusenehas/Fraud-detector.git



Install Dependencies:
pip install -r requirements.txt



Run the Streamlit App:
streamlit run app.py
<br><br><br><br>
<i>Last Updated: August 11, 2025</i>
