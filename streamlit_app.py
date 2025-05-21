import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load('house_price_model.pkl')

# Title and intro
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.markdown("""
This app uses smart regression techniques to forecast **house prices** based on user-input features. 
Trained with **Random Forest and Linear Models** for high accuracy.
""")

# Sidebar for input
st.sidebar.header("Enter Property Features")

def user_input_features():
    LB = st.sidebar.slider('Living Area (LB) in sqm', 50, 500, 120)
    LT = st.sidebar.slider('Land Area (LT) in sqm', 50, 1000, 200)
    KT = st.sidebar.slider('Number of Bedrooms (KT)', 1, 10, 3)
    KM = st.sidebar.slider('Number of Bathrooms (KM)', 1, 10, 2)
    GRS = st.sidebar.selectbox('Garage (GRS)', [0, 1, 2, 3])
    data = {'LB': LB, 'LT': LT, 'KT': KT, 'KM': KM, 'GRS': GRS}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Predict
if st.button("Predict House Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Estimated House Price: ₹{prediction[0]:,.2f}")

# Visual Insights (optional EDA visualization)
st.markdown("### 🔍 Model Insights")

if st.checkbox("Show Feature Importance"):
    try:
        importances = model.feature_importances_
        features = ['LB', 'LT', 'KT', 'KM', 'GRS']
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=features, ax=ax)
        ax.set_title("Feature Importance in House Price Prediction")
        st.pyplot(fig)
    except:
        st.warning("Feature importances not available for this model.")

# About Section
st.markdown("### 👥 Team Members & Contributions")
st.markdown("""
- **Surender**: Problem Statement and Project Objectives  
- **Sridhar**: Workflow, Data Description, EDA  
- **Sunil**: Feature Engineering and Model Building  
- **Vathish**: Results Visualization & Technologies  
""")

st.markdown("📂 [View GitHub Repository](https://github.com/SUNIL065/project.git)")

