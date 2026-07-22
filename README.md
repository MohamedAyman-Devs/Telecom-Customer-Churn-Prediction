# 📊 Telecom Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to leave the company (Churn) based on customer demographics, account information, and subscribed services.

---

## 📖 Overview

Customer churn prediction is one of the most common classification problems in the telecom industry. This project applies data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning, and deployment to predict customer churn.

---

## 🎯 Objectives

- Clean and preprocess customer data.
- Perform Exploratory Data Analysis (EDA).
- Apply Feature Engineering.
- Train and evaluate a Machine Learning model.
- Deploy the model using Streamlit.

---

## 📂 Project Structure

```
Telecom-Customer-Churn-Prediction
│
├── app/
│   └── app.py
│
├── data/
│   ├── TelecomCustomerChurn.csv
│   └── cleaned_dataset.csv
│
├── images/
│
├── models/
│   └── logistic_regression_model.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── FeatureEng.ipynb
│   └── Deployment.ipynb
│
├── src/
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

---

## 📊 Dataset

The dataset contains customer information including:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Contract Type
- Monthly Charges
- Total Charges
- Payment Method
- Churn

---

## 🤖 Machine Learning

Current Model:

- Logistic Regression

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🚀 Running the Project

Clone the repository

```bash
git clone https://github.com/MohamedAyman-Devs/Telecom-Customer-Churn-Prediction.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app/app.py
```

---

## 📌 Future Improvements

- Add more Machine Learning models.
- Hyperparameter Tuning.
- Improve Streamlit UI.
- Deploy online.
- Add Model Explainability (SHAP).

---

## 👨‍💻 Author

**Mohamed Ayman**

- LinkedIn: www.linkedin.com/in/mohamed-ayman-2036413a5
- GitHub: https://github.com/MohamedAyman-Devs
