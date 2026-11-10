# Customer Churn Prediction System

## Project Overview
This project uses Machine Learning to predict whether a customer is likely to leave a company. It uses the Telco Customer Churn dataset and a Logistic Regression classification model.

## Features
- Customer churn prediction
- Data preprocessing and feature encoding
- Logistic Regression model training
- Model evaluation
- Prediction probability
- Streamlit web application

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn

## Project Files
- `app.py` – Streamlit web application
- `customer_churn_model.pkl` – Trained machine learning model
- `model_columns.pkl` – Model feature columns

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Machine Learning Model
The project uses Logistic Regression to classify customers into two categories:
- Likely to Stay
- May Churn

## Dataset
IBM Telco Customer Churn dataset.

## Author
Ubaid Ullah
