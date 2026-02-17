# 🌍 Tourism Experience Analytics and Rating Prediction using Machine Learning

## Project Overview
Tourism Experience Analytics is a Machine Learning–based application designed to analyze tourist visit data and predict **tourist rating categories** based on travel and location characteristics.  
The system helps tourism stakeholders understand factors influencing visitor satisfaction and enables data-driven decision making.

The project follows a complete **end-to-end ML pipeline**, including data preprocessing, feature engineering, model training, evaluation, and deployment using **Streamlit**.


## Objectives
- Analyze tourism transaction data to extract meaningful insights
- Predict tourist rating categories (Low, Medium, High)
- Deploy a user-friendly web application for real-time predictions
- Maintain consistency between training and inference pipelines


## Machine Learning Pipeline
1. **Data Collection & Cleaning**
   - Multiple tourism-related datasets merged and cleaned
   - Missing values handled and data standardized

2. **Feature Engineering**
   - Categorical encoding using `pd.get_dummies()`
   - Feature selection based on relevance
   - Train–test split with stratification

3. **Feature Scaling**
   - `StandardScaler` applied to numerical features

4. **Model Training**
   - Models evaluated: Logistic Regression, Random Forest
   - **Gradient Boosting Classifier** selected due to superior performance

5. **Model Evaluation**
   - Accuracy and classification metrics
   - Model validated for generalization

6. **Deployment**
   - Streamlit web application
   - Same preprocessing logic applied during inference

## Best Performing Model
- **Algorithm:** Gradient Boosting Classifier  
- **Why Gradient Boosting?**
  - Handles non-linear relationships well
  - Robust to feature interactions
  - Performs well on imbalanced datasets



## Web Application (Streamlit)
The Streamlit app provides:
- Interactive UI for entering trip details
- Real-time rating prediction
- Multiple pages:
  - Home
  - Predict Rating
  - Data Insights
  - About

### Prediction Inputs
- Attraction Type
- Visit Mode
- Season
- Continent
- Region
- Country
- City

### Output
- ⭐ Low Rating  
- ⭐⭐ Medium Rating  
- ⭐⭐⭐ High Rating  



## Project Structure
Tourism_Experience_Analytics/
│
├── app/
│ └── app.py # Streamlit application
│
├── notebook/
│ └── eda_ml.ipynb # EDA & ML pipeline
│
├── data/
│ └── raw/ # Raw datasets (ignored in Git)
│
├── models/ # Trained models (ignored in Git)
│
├── requirements.txt
├── .gitignore
└── README.md



> ⚠️ Note:  
> Trained models and raw datasets are excluded from version control due to size constraints and reproducibility considerations.


## Installation & Setup

###  Clone the Repository
```bash
git clone https://github.com/Aadhithyan-2005/Tourism-Experience-Analytics.git
cd Tourism-Experience-Analytics


## Create virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

## Install Dependencies
pip install -r requirements.txt

## Running the application
streamlit run app/app.py

Then open the browser at
http://localhost:8501

## Technologies Used

-Python

-Pandas & NumPy

-Scikit-Learn

-Streamlit

-Matplotlib & Seaborn

-Joblib

-Git & GitHub

## Academic Relevance

This project demonstrates:

End-to-end ML workflow

Proper feature engineering & preprocessing

Deployment-ready ML design

Clean version control practices

Suitable for:

Final Year Project (B.Tech / B.E.)

ML & Data Science portfolios

Academic evaluations and viva


**Author**
Aadhithyan M



