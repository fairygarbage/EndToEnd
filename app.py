import streamlit as st  
import pandas as pd    
import numpy as np     
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline          
from sklearn.impute import SimpleImputer        
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder  
from sklearn.base import BaseEstimator, TransformerMixin  
import joblib      

def predicts(data):
    model = joblib.load("final_model.pkl")
    full_pipeline = fetch_pipeline()
    data_prepared = full_pipeline.transform(data)
    
    return model.predict(data_prepared)

def fetch_pipeline():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),  
        ('attribs_adder', CombinedAttributesAdder()),  
        ('std_scaler', StandardScaler())                
    ])

    housing = pd.read_csv("datasets housing/housing.csv")
    
    housing = housing.drop("median_house_value", axis=1)
    
    housing_num = housing.drop("ocean_proximity", axis=1)
    
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),   
        ("cat", OneHotEncoder(), cat_attribs) 
    ])
    
    full_pipeline.fit(housing)

    return full_pipeline

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

st.title("Prediction Model")
with st.container():
    longitude = st.number_input("Longitude", max_value=0.0)
    latitude = st.number_input("Latitude", min_value=1.0)
    population = st.number_input("Population", min_value=1.0)

with st.container():
    total_rooms = st.number_input("Total rooms", min_value=1.0)
    total_bedrooms = st.number_input("Total bedrooms", min_value=1.0)
    households = st.number_input("Households", min_value=1.0)
    age = st.number_input("Housing Median Age", min_value=1.0)
    income = st.number_input("Median income", min_value=1.0)

ocean_proximity = st.selectbox("Ocean Proximity: ",['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

if st.button("Predict house value"):    
    data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [income],
        'ocean_proximity': [ocean_proximity]
    })
    
    result = predicts(data)
    st.text(result[0])
