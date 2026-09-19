# 📊 Customer Churn Prediction

An end-to-end Machine Learning application that predicts whether a customer is likely to churn based on demographic information, services, contract details, and billing information.

The project demonstrates a complete ML workflow from **data preprocessing and model training to prediction and interactive deployment using Streamlit**.

---

## 🚀 Project Overview

Customer churn is an important business problem where companies try to identify customers who are likely to stop using their services.

This project uses customer information to predict:

- `Yes` → Customer is likely to churn
- `No` → Customer is likely to stay

The trained Machine Learning pipeline is integrated into a Streamlit web application where users can enter customer information and receive a prediction with the estimated churn probability.

---

## 🧠 Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature / Target Separation
     │
     ▼
Train / Test Split
     │
     ▼
Preprocessing
 ┌───┴───────────────┐
 │                   │
 ▼                   ▼
Numerical          Categorical
Features           Features
 │                   │
 ▼                   ▼
StandardScaler   OneHotEncoder
 └───────┬───────────┘
         ▼
   Logistic Regression
         │
         ▼
     Evaluation
         │
         ▼
    Save Model
         │
         ▼
   Streamlit App
         │
         ▼
   Churn Prediction
````

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* StandardScaler
* OneHotEncoder
* ColumnTransformer
* Pipeline

### Model Serialization

* Joblib

### Deployment / Web App

* Streamlit

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains customer information related to:

* Demographics
* Phone services
* Internet services
* Online services
* Contract type
* Payment method
* Monthly charges
* Total charges
* Customer churn

### Dataset

[Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🧹 Data Preprocessing

The dataset required several preprocessing steps before training.

### 1. Convert `TotalCharges`

`TotalCharges` was originally stored as an object/string column.

It was converted to numeric values:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Invalid values were converted to `NaN`.

### 2. Remove Missing Values

Rows with missing `TotalCharges` values were removed:

```python
df = df.dropna(
    subset=["TotalCharges"]
)
```

### 3. Remove Customer ID

`customerID` is an identifier and does not provide useful predictive information:

```python
df = df.drop(
    columns=["customerID"]
)
```

### 4. Numerical Features

The following numerical features were standardized using `StandardScaler`:

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

### 5. Categorical Features

Categorical features were converted using `OneHotEncoder`:

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

Using `handle_unknown="ignore"` allows the model to handle previously unseen categorical values during prediction.

---

## 🤖 Machine Learning Model

The current baseline model is:

### Logistic Regression

The model is implemented inside a Scikit-learn `Pipeline`:

```python
model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])
```

This approach keeps preprocessing and prediction together in one reusable object.

---

## 🔄 Why Use a Pipeline?

Instead of manually preprocessing the training data and then repeating the same preprocessing during prediction, the entire workflow is saved together.

```text
Raw Input
   ↓
Preprocessor
   ↓
Logistic Regression
   ↓
Prediction
```

The complete pipeline is saved using Joblib:

```python
joblib.dump(
    model,
    "models/churn_model.pkl"
)
```

This ensures that the same preprocessing logic is used during both training and inference.

---

## 📈 Model Performance

The current Logistic Regression model was evaluated on a held-out test set.

### Results

| Metric          |  Score |
| --------------- | -----: |
| Accuracy        | 80.38% |
| Precision — No  |   0.85 |
| Recall — No     |   0.89 |
| F1 — No         |   0.87 |
| Precision — Yes |   0.65 |
| Recall — Yes    |   0.57 |
| F1 — Yes        |   0.61 |

### Classification Report

```text
              precision    recall  f1-score   support

          No       0.85      0.89      0.87      1033
         Yes       0.65      0.57      0.61       374

    accuracy                           0.80      1407
   macro avg       0.75      0.73      0.74      1407
weighted avg       0.80      0.80      0.80      1407
```

> The current model is a baseline implementation. Further improvements can include model comparison, hyperparameter tuning, class-imbalance techniques, and threshold optimization.

---

## 🔮 Prediction

The project also contains a standalone prediction script:

```text
src/predict.py
```

It loads the trained pipeline:

```python
model = joblib.load(
    "models/churn_model.pkl"
)
```

Then creates a new customer's input as a Pandas DataFrame and sends it directly to the pipeline.

Example:

```text
Prediction: Yes
```

---

## 🌐 Streamlit Application

The project includes an interactive web application built with Streamlit.

The application allows users to enter:

### 👤 Customer Information

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure

### 📡 Services

* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

### 💳 Contract & Billing

* Contract
* Paperless Billing
* Payment Method

### 💰 Charges

* Monthly Charges
* Total Charges

The application returns:

```text
Prediction
+
Churn Probability
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
```

```bash
cd customer-churn-prediction
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train the Model

```powershell
python src/train.py
```

This will:

1. Load the dataset
2. Clean the data
3. Split the dataset
4. Build the preprocessing pipeline
5. Train Logistic Regression
6. Evaluate the model
7. Save the trained pipeline

The trained model will be saved to:

```text
models/churn_model.pkl
```

---

### 🔮 Run Prediction Script

```powershell
python src/predict.py
```

---

### 🌐 Run Streamlit Application

```powershell
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 💡 Key Concepts Demonstrated

This project demonstrates several important Machine Learning engineering concepts:

* Data cleaning
* Feature/target separation
* Train/test splitting
* Numerical preprocessing
* Categorical encoding
* Scikit-learn pipelines
* Logistic Regression
* Model evaluation
* Model serialization
* Batch inference
* Interactive ML applications
* Streamlit deployment
* Project organization
* Virtual environments
* Git/GitHub workflow

---

## 🔮 Future Improvements

* [ ] Exploratory Data Analysis
* [ ] Compare Logistic Regression, Random Forest, and XGBoost
* [ ] Hyperparameter tuning
* [ ] Cross-validation
* [ ] Class imbalance handling
* [ ] Confusion matrix visualization
* [ ] Feature importance analysis
* [ ] Model versioning
* [ ] FastAPI backend
* [ ] Dockerization
* [ ] CI/CD pipeline
* [ ] Cloud deployment

---

## 👨‍💻 Author

**Ahmed Ragab**

Computer Science & Information Student — AI Department

### Interests

* Machine Learning
* Data Analysis
* AI Applications
* ML Engineering

---

## ⭐ Project Goal

The main goal of this project is to demonstrate how a Machine Learning model can move beyond a Jupyter Notebook into a structured, reusable, and deployable application.

```text
Notebook
   ↓
Python Scripts
   ↓
ML Pipeline
   ↓
Saved Model
   ↓
Prediction
   ↓
Streamlit Application
```

```

ده جاهز للنسخ مباشرة إلى `README.md`.
```
