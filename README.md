# ML_Projects
This repository contain Projects predicting Employee Attribution, Text Summarization project, Disease Diagnosis Prediction and Default Loan Prediction . 


# 📊 Predicting Employee Attrition with IBM HR Analytics Dataset

## Overview
This project aims to predict employee attrition using the IBM HR Analytics Employee Attrition & Performance dataset. It employs various machine learning models to identify employees at risk of leaving, provides model interpretability through SHAP and LIME, and offers actionable retention strategies. An interactive Plotly dashboard visualizes key attrition patterns and model insights for HR decision support.

## Features
- **Data Preprocessing**: Handles categorical encoding, numerical scaling, and feature engineering (e.g., dropping irrelevant columns like `EmployeeNumber`).
- **Model Training**: Evaluates multiple classifiers (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) with class imbalance handling.
- **Hyperparameter Tuning**: Optimizes the best model using GridSearchCV with F1-score as the primary metric.
- **Model Explainability**: Uses SHAP for global feature importance and LIME for individual prediction explanations.
- **Exploratory Data Analysis (EDA)**: Visualizes attrition patterns using Seaborn, Matplotlib, and Plotly.
- **Interactive Dashboard**: Provides a Plotly dashboard with attrition by department, age vs. income, SHAP importance, and risk distribution.
- **Actionable Insights**: Recommends six retention strategies targeting overtime, job satisfaction, compensation, and more.

## Dataset
The project uses the **IBM HR Analytics Employee Attrition & Performance Dataset** (1470 samples, 35 features), containing employee attributes such as age, income, job satisfaction, and attrition status (Yes/No). The dataset is publicly available and included in the project repository as `WA_Fn-UseC_-HR-Employee-Attrition.csv`.





#📊  Text Summarization with CNN/Daily Mail Dataset

## Overview
This project implements text summarization techniques using the CNN/Daily Mail dataset (version 3.0.0). It includes both extractive summarization using spaCy and abstractive summarization with a fine-tuned T5 model. The project evaluates the performance of the fine-tuned model using ROUGE scores and applies it to real-world articles for summarization.

## Features
- **Extractive Summarization**: Uses spaCy to select key sentences based on word frequency.
- **Abstractive Summarization**: Fine-tunes a T5 model for generating concise summaries.
- **Evaluation**: Computes ROUGE-1, ROUGE-2, and ROUGE-L scores to assess summary quality.
- **Real-world Application**: Generates summaries for sample articles using the fine-tuned T5 model.

## Dataset
The project uses the CNN/Daily Mail dataset (`cnn_dailymail`, version 3.0.0) from Hugging Face, which contains news articles and their corresponding highlights (summaries). A subset of 1000 training samples and 100 validation samples is used for faster processing.


#📊  Disease Diagnosis Prediction (Diabetes)

## Overview
This project predicts the likelihood of diabetes using the PIMA Indian Diabetes Dataset. It implements a machine learning pipeline with exploratory data analysis (EDA), feature selection, handling class imbalance, model training, and evaluation. The project provides healthcare insights and includes a function for assessing patient risk, making it a valuable tool for medical decision support.

## 🎯Features
- **Exploratory Data Analysis (EDA)**: Visualizes data distributions, correlations, and class imbalance using Seaborn and Matplotlib.
- **Feature Selection**: Uses SelectKBest (chi-squared) and Recursive Feature Elimination (RFE) to identify key predictors.
- **Class Imbalance Handling**: Applies Synthetic Minority Oversampling Technique (SMOTE) to balance the dataset.
- **Model Training**: Evaluates Gradient Boosting, Support Vector Machine (SVM), and Neural Network models.
- **Hyperparameter Tuning**: Optimizes the best model (Neural Network) using GridSearchCV.
- **Model Evaluation**: Reports F1 Score, AUC-ROC, precision, recall, and confusion matrix.
- **Healthcare Insights**: Provides actionable recommendations for diabetes risk management.
- **Patient Risk Assessment**: Includes a function to predict diabetes risk for new patients.
- **Visualization**: Generates plots for feature importance, ROC curves, and confusion matrices.

## Dataset
The project uses the **PIMA Indian Diabetes Dataset**, which contains 768 samples with 8 features (e.g., Glucose, BMI, Age, Insulin) and a binary target variable (Outcome: 0 = No Diabetes, 1 = Diabetes). The dataset is publicly available and can be obtained from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database) or included in the repository as `diabetes.csv`.



# 📊 Loan Default Prediction - Task 4

This project focuses on building a robust machine learning pipeline to predict loan default using a synthetic Lending Club-style dataset. It includes the complete data science workflow, from Exploratory Data Analysis (EDA) to model saving and risk assessment, to support lenders in making informed decisions.

---

## 🗂 Project Structure


---

## 🎯 Problem Statement

The objective is to build a classification model that predicts whether a loan will be paid back or defaulted based on applicant and loan-related features.

---

## 📁 Dataset Description

- **Type:** Synthetic Lending Club-style dataset  
- **Target variable:** `loan_status` (Default or Paid)  
- **Features include:** applicant income, credit history, loan purpose, employment status, etc.

---

## ⚙️ Methodology

### 🔍 1. Exploratory Data Analysis (EDA)
- Analyzed distributions, default rates, correlations.
- Identified data imbalances and feature trends.

### 🧹 2. Data Preprocessing
- **Numerical Features:** Scaled using `StandardScaler`.
- **Categorical Features:** Encoded using `OneHotEncoder`.
- **Class Imbalance:** Resolved using `SMOTE`.

### 🤖 3. Model Training
Trained and compared the performance of the following models:
- **LightGBM**
- **Random Forest**
- **Support Vector Machine (SVM)**

### 📊 4. Model Evaluation
Evaluated all models using:
- **F1 Score**
- **AUC-ROC Curve**
- **Confusion Matrix**

### 🏆 Best Model: LightGBM
- **Tuned F1 Score:** `0.851`
- **AUC-ROC:** `0.958`

---

## 💾 Saved Artifacts

Artifacts are stored in the `loan_default_model/` directory:

| File                    | Description                                  |
|-------------------------|----------------------------------------------|
| `best_model.pkl`        | Trained LightGBM model                       |
| `scaler.pkl`            | StandardScaler for numerical features        |
| `encoder.pkl`           | OneHotEncoder for categorical features       |
| `feature_columns.json`  | Feature names used for model training        |

---

## 🌟 Key Features

- ✅ **Risk Assessment Function** to predict loan default for new applicants.
- 📈 **Feature Importance Analysis** using SHAP and model internals.
- 📊 **Lender Performance Report** to evaluate loan outcomes per lender.

---

