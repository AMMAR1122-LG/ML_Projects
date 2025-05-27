# ML_Projects
This repository contain Projects predicting Employee Attribution, Text Summarization project, Disease Diagnosis Prediction and Default Loan Prediction . 


# Predicting Employee Attrition with IBM HR Analytics Dataset

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





# Text Summarization with CNN/Daily Mail Dataset

## Overview
This project implements text summarization techniques using the CNN/Daily Mail dataset (version 3.0.0). It includes both extractive summarization using spaCy and abstractive summarization with a fine-tuned T5 model. The project evaluates the performance of the fine-tuned model using ROUGE scores and applies it to real-world articles for summarization.

## Features
- **Extractive Summarization**: Uses spaCy to select key sentences based on word frequency.
- **Abstractive Summarization**: Fine-tunes a T5 model for generating concise summaries.
- **Evaluation**: Computes ROUGE-1, ROUGE-2, and ROUGE-L scores to assess summary quality.
- **Real-world Application**: Generates summaries for sample articles using the fine-tuned T5 model.

## Dataset
The project uses the CNN/Daily Mail dataset (`cnn_dailymail`, version 3.0.0) from Hugging Face, which contains news articles and their corresponding highlights (summaries). A subset of 1000 training samples and 100 validation samples is used for faster processing.





   **Disease Diagnosis Prediction (Diabetes)**
    -   Uses the PIMA Indian Diabetes Dataset to predict the likelihood of diabetes.
    -   Employs EDA, feature selection (SelectKBest, RFE), SMOTE for imbalance, and trains Gradient Boosting, SVM, and Neural Network models.
    -   **Best Model:** Neural Network (F1 Score: 0.802, AUC-ROC: 0.836 after tuning).
    -   Provides healthcare insights and a function for patient risk assessment.
    -   Notebook: `Task_3.ipynb`

  **Loan Default Prediction**
    -   Uses a synthetic Lending Club-style dataset to predict loan default.
    -   Includes EDA, preprocessing (StandardScaler, OneHotEncoder), SMOTE, and trains LightGBM, SVM, and Random Forest models.
    -   **Best Model:** LightGBM (Tuned F1 Score: 0.851, AUC-ROC: 0.958).
    -   Features model saving, a loan risk assessment function, feature importance analysis, and a lender performance report.
    -   Notebook: `Task_4.ipynb`
    -   Artifacts saved in: `loan_default_model/`

