import os
import pickle

import streamlit as st
from dotenv import load_dotenv

from utils.b2 import B2
from utils.modeling import *

import pandas as pd
import altair as alt
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and TF-IDF vectorizer
@st.cache(allow_output_mutation=True)
def load_model():
    with open('model.pickle', 'rb') as model_file, open('tfidf_vectorizer.pickle', 'rb') as tfidf_file:
        model = pickle.load(model_file)
        tfidf_vectorizer = pickle.load(tfidf_file)
    return model, tfidf_vectorizer

model, tfidf_vectorizer = load_model()

# Streamlit page configuration
st.title("Cosmetic Product Rating Predictor")

# User input for ingredients
ingredients_input = st.text_area("Enter the ingredients of the cosmetic product:", height=150)

# Predict button
if st.button('Predict Rating'):
    # Transform the input to the same format as the model was trained on
    transformed_input = tfidf_vectorizer.transform([ingredients_input])
    # Prediction
    prediction = model.predict(transformed_input)
    # Display the prediction
    st.write(f"Predicted Rating: {prediction[0]:.2f}")

# Optionally, add more interactive or informative elements as needed
