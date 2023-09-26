import streamlit as st  
import pandas as pd    
import numpy as np     
import joblib  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline          
from sklearn.impute import SimpleImputer        
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder  
from sklearn.base import BaseEstimator, TransformerMixin 
import joblib

def predict(data):

    model = joblib.load('final_model.pkl')

    pipeline = joblib.load("full_pipeline.pkl")
    data = pipeline.transform(data)
    
    return model.predict(data)
    
st.header('House prediction base in Californa Prices Values DataSet')
st.write('Data Science Project')

col1, col2, col3 = st.columns(3)

with st.container():
    st.write("Indique la locación")
    longitude = col1.number_input('Longitude', min_value = -124.0, format = "%.2f")
    latitude = col1.number_input('Latitude', min_value = 30.0)
    population = col1.number_input('Population', min_value = 1.0, max_value = 50000.0, format = "%.0f")
    ocean_proximity = col1.selectbox('Proximity to ocean', ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])


with st.container():
    total_rooms = col2.number_input('Total de habitaciones', min_value = 1.0, max_value = 50000.0, format = "%.0f")
    total_bedrooms = col2.number_input('Total de dormitorios', min_value = 1.0, max_value = 7000.0, format = "%.0f")

with st.container():
    households = col3.number_input('Households', min_value = 1.0, max_value = 10000.0, format = "%.0f")
    housing_median_age = col3.slider('Housing median age', step=1.0, min_value=1.0, max_value=100.0, value=0.0, format = "%.0f")
    median_income = col3.number_input('Median income', min_value = 0.0, max_value = 17.0, format = "%.4f")

    if st.button('Search prediction'):
        data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'ocean_proximity': [ocean_proximity]}
        )

        result = prediction.predict(data)
        st.write("The predicted value is of {:.1f} usd".format(result[0]))
