import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Tourism Experience Analytics and Rating Prediction Using Machine Learning",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 { color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# NOTE:

DATA_AVAILABLE = False  


@st.cache_resource
def load_models():
    model = joblib.load(os.path.join(MODELS_DIR, "gradient_boosting_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    model_columns = joblib.load(os.path.join(MODELS_DIR, "model_columns.joblib"))
    scaler_columns = joblib.load(os.path.join(MODELS_DIR, "scaler_columns.joblib"))
    return model, scaler, model_columns, scaler_columns


model, scaler, model_columns, scaler_columns = load_models()
st.write("Scaler expects features:", scaler.n_features_in_)
st.write("Scaler columns count:", len(scaler_columns))
st.write("Model columns count:", len(model_columns))

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2060/2060284.png", width=80)
    st.title("Navigation")
    page = st.radio("Go to", ["Home", "Predict Rating", "Data Insights", "ℹ️ About"])
    st.markdown("---")
    st.caption("Developed by **Aadhithyan M**")
    st.caption("Powered by Streamlit & Scikit-Learn")


if page == "Home":
    st.title("🌍 Tourism Experience Analytics")
    st.markdown("### Optimize Tourist Satisfaction with AI")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("""
        **Project Goal:**  
        This application leverages **Machine Learning** to predict tourist rating categories 
        based on travel and location characteristics.
        """)

        st.markdown("#### Key Features")
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("**Instant Predictions**\n\nReal-time rating forecasting.")
        with c_b:
            st.markdown("**Interactive EDA**\n\nInsights from historical tourism data.")

    with col2:
        st.markdown(
            '<div class="metric-card"><h3> Best Model</h3><p>Gradient Boosting Classifier</p></div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-card"><h3> Dataset</h3><p>52,000+ Records</p></div>',
            unsafe_allow_html=True
        )

elif page == "Predict Rating":
    st.title("Rating Prediction Engine")
    st.markdown("Configure the trip details below to predict the **rating category**.")

    with st.form("predict_form"):
        st.subheader("1Trip Characteristics")
        c1, c2, c3 = st.columns(3)

        with c1:
            attraction_type = st.selectbox(
                "Attraction Type",
                ["Beach", "Historical", "Religious", "Adventure", "Nature", "Urban", "Shopping"]
            )

        with c2:
            visit_mode = st.selectbox(
                "Visit Mode",
                ["Solo", "Family", "Friends", "Business", "Couples"]
            )

        with c3:
            season = st.selectbox(
                "Season",
                ["Summer", "Winter", "Monsoon", "Spring"]
            )

        st.subheader("Location Context")
        c4, c5 = st.columns(2)

        with c4:
            continent = st.selectbox(
                "Continent",
                ["Asia", "Europe", "North America", "South America", "Africa", "Australia"]
            )

            region = st.selectbox(
                "Region",
                [
                    "South Asia", "Western Europe", "Northern America",
                    "Central Europe", "East Asia", "Other"
                ]
            )

        with c5:
            country = st.selectbox(
                "Country",
                ["India", "United States", "United Kingdom", "France", "Germany", "Australia", "Other"]
            )

            city = st.selectbox(
                "City",
                ["Mumbai", "New York", "London", "Paris", "Berlin", "Sydney", "Other"]
            )

        submit = st.form_submit_button("Generate Prediction")

    if submit:
        with st.spinner("Analyzing parameters..."):
            time.sleep(0.5)

            input_df = pd.DataFrame([{
                "AttractionType": attraction_type,
                "VisitMode": visit_mode,
                "Season": season,
                "Continent": continent,
                "Region": region,
                "Country": country,
                "CityName": city
            }])

            input_encoded = pd.get_dummies(
                input_df,
                drop_first=True,
                dtype=int
            )

            input_encoded = input_encoded.reindex(
                columns=model_columns,
                fill_value=0
            )

            input_scaled = scaler.transform(input_encoded[scaler_columns])


            prediction = model.predict(input_scaled)[0]

            st.markdown("---")
            if prediction == 2:
                st.balloons()
                st.success("### ⭐⭐⭐ High Rating\n**Excellent Tourist Experience**")
            elif prediction == 1:
                st.warning("### ⭐⭐ Medium Rating\n**Average Tourist Experience**")
            else:
                st.error("### ⭐ Low Rating\n**Poor Tourist Experience**")

elif page == "Data Insights":
    st.title("Exploratory Data Analysis")

    st.info("""
    The EDA visualizations were performed during model development.
    To avoid inconsistencies, raw data is not reloaded here.
    """)

    st.markdown("""
    **Key Insights from Training Data:**
    - Ratings are skewed towards positive experiences
    - Nature and Historical attractions dominate visits
    - Seasonal patterns influence tourist satisfaction
    """)

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ### Project Overview
    This system predicts **tourist rating categories** using supervised machine learning.
    The pipeline includes data preprocessing, one-hot encoding, feature scaling, 
    and Gradient Boosting classification.

    ### Model Selection
    Multiple models were evaluated, and **Gradient Boosting** was chosen for its
    superior performance on imbalanced tourism data.

    ### Tech Stack
    - Python, Pandas, NumPy  
    - Scikit-Learn  
    - Streamlit  
    - Seaborn & Matplotlib  
    """)

    st.markdown("---")
