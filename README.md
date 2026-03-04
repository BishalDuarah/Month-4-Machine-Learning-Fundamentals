# Month-4-Machine-Learning-Fundamentals
# 📊 Customer Churn Prediction System

An **end-to-end Machine Learning system** that predicts telecom customer churn using a tuned **Random Forest model** and exposes predictions through a **FastAPI REST API**.

This project demonstrates the **complete ML lifecycle**, including data preprocessing, feature engineering, model training, evaluation, explainability, and deployment.

---

# 🚀 Project Overview

Customer churn is a critical problem in subscription-based businesses like telecom companies.
This system predicts whether a customer is likely to churn based on their service usage, contract type, billing information, and support features.

The goal is to help companies **identify high-risk customers and take proactive retention actions.**

---

# 🧠 Machine Learning Workflow

```
Raw Dataset
     │
     ▼
Data Cleaning & Preprocessing
     │
     ▼
Feature Engineering Pipeline
     │
     ▼
Model Training (3 Algorithms)
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Model Evaluation & Feature Importance
     │
     ▼
Model Serialization (Joblib)
     │
     ▼
Inference Pipeline
     │
     ▼
FastAPI REST API
```

---

# 🏗 System Architecture

```
                   ┌───────────────────────┐
                   │   Kaggle Dataset      │
                   │ Telco Customer Churn │
                   └─────────────┬─────────┘
                                 │
                                 ▼
                     ┌─────────────────────┐
                     │ Data Preprocessing  │
                     │ Cleaning + Encoding │
                     └─────────────┬───────┘
                                   │
                                   ▼
                         ┌──────────────────┐
                         │ Feature Pipeline │
                         │ ColumnTransformer│
                         └──────────┬───────┘
                                    │
                                    ▼
                         ┌──────────────────┐
                         │ Machine Learning │
                         │ Random Forest    │
                         └──────────┬───────┘
                                    │
                                    ▼
                         ┌──────────────────┐
                         │ Saved Model      │
                         │ .pkl artifacts   │
                         └──────────┬───────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │ FastAPI Server │
                           │ /predict API   │
                           └────────┬───────┘
                                    │
                                    ▼
                           JSON Prediction
```

---

# 📁 Project Structure

```
Month-4 Machine Learning Fundamentals/
│
├── data/
│   └── raw/
│       └── telco_churn.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_test.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_inference_test.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_inference.py
│   └── web_app.py
│
├── models/
│   ├── random_forest_model.pkl
│   └── preprocessor.pkl
│
├── docs/
│   ├── problem_statement.md
│   └── model_interpretation.md
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

Dataset used:

**Telco Customer Churn Dataset (IBM)**

Features include:

* Customer demographics
* Contract type
* Internet services
* Tech support
* Billing information
* Customer lifetime value

Target variable:

```
Churn Value
0 = Customer stays
1 = Customer churns
```

---

# ⚙️ Machine Learning Models

Three algorithms were implemented and compared:

| Model               | Purpose                      |
| ------------------- | ---------------------------- |
| Logistic Regression | Baseline interpretable model |
| Decision Tree       | Nonlinear baseline           |
| Random Forest       | Final production model       |

The **Random Forest model** performed best after hyperparameter tuning.

---

# 📈 Final Model Performance

| Metric    | Score    |
| --------- | -------- |
| Accuracy  | 0.77     |
| Precision | 0.55     |
| Recall    | 0.73     |
| F1 Score  | 0.63     |
| ROC-AUC   | **0.85** |

---

# 🔍 Top Feature Importance

The model identified the following key churn drivers:

1. **Tenure Months**
2. **Contract (Month-to-month)**
3. **Monthly Charges**
4. **Total Charges**
5. **Tech Support Availability**

These insights align with known telecom churn behavior.

---

# ⚡ API Deployment (FastAPI)

The trained model is deployed through a **FastAPI REST API**.

### Run the server

```bash
python -m uvicorn src.web_app:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

---

# 📘 API Documentation

Interactive Swagger UI:

```
http://127.0.0.1:8000/docs
```

Example request:

```json
{
  "Gender": "Female",
  "Senior_Citizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "Tenure_Months": 5,
  "Phone_Service": "Yes",
  "Multiple_Lines": "No",
  "Internet_Service": "Fiber optic",
  "Online_Security": "No",
  "Online_Backup": "No",
  "Device_Protection": "No",
  "Tech_Support": "No",
  "Streaming_TV": "Yes",
  "Streaming_Movies": "Yes",
  "Contract": "Month-to-month",
  "Paperless_Billing": "Yes",
  "Payment_Method": "Electronic check",
  "Monthly_Charges": 85,
  "Total_Charges": 425,
  "CLTV": 3000
}
```

Example response:

```json
{
  "churn_probability": 0.78,
  "churn_prediction": 1
}
```

---

# 🛠 Tech Stack

* Python
* Pandas
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib
* Matplotlib

---

# 🎯 Key Features

✔ End-to-end ML pipeline
✔ Feature engineering pipeline
✔ Multiple model comparison
✔ Hyperparameter tuning with GridSearchCV
✔ Model explainability with feature importance
✔ Production-ready FastAPI deployment
✔ Modular project structure

---

# 👨‍💻 Author

**Bishal Duarah**
B.Tech CSE | Data Science & Machine Learning Enthusiast

---

# ⭐ Future Improvements

* Deploy API to cloud (Render / AWS / Railway)
* Add a web dashboard for predictions
* Implement SHAP model explainability
* Build real-time data ingestion pipeline
