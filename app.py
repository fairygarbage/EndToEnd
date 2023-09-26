import streamlit as st  
import pandas as pd    
import numpy as np     
import joblib  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline          
from sklearn.impute import SimpleImputer        
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder  
from sklearn.base import BaseEstimator, TransformerMixin 

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self,X, y = None):
        return self

    def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
            population_per_household = X[:, population_ix] / X[:, household_ix]
            if self.add_bedrooms_per_room:
                    bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                    return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]


def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
            population_per_household = X[:, population_ix] / X[:, household_ix]
            if self.add_bedrooms_per_room:
                    bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                    return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

def predicts(data):
    model = joblib.load("final_model.pkl")
    pipeline = joblib.load("full_pipeline.pkl")
    data = pipeline.transform(data)
    
    return model.predicts(data)
    

st.header('Model Prediction house price value in California')

longitude = st.number_input("Longitude", max_value=0.0)
latitude = st.number_input("Latitude", min_value=1.0)
housing_median_age = st.number_input("Housing Median Age", min_value=1.0)
total_rooms = st.number_input("Total rooms", min_value=1.0)
total_bedrooms = st.number_input("Total bedrooms", min_value=1.0)
population = st.number_input("Population", min_value=1.0)
households = st.number_input("Households", min_value=1.0)
income = st.number_input("Median income", min_value=1.0)
ocean_proximity = st.selectbox("Ocean Proximity: ",['<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN', 'INLAND', 'ISLAND'])

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

result = predicts(data)
st.write("The predicted value is {:.1f} USD".format(result[0]))
