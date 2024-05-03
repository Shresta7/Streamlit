import streamlit as st
import pandas as pd
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

@st.cache(allow_output_mutation=True)
def load_model_and_vectorizer():
    with open('model.pickle', 'rb') as model_file, open('tfidf_vectorizer.pickle', 'rb') as tfidf_file:
        model = pickle.load(model_file)
        tfidf_vectorizer = pickle.load(tfidf_file)
    return model, tfidf_vectorizer

model, tfidf_vectorizer = load_model_and_vectorizer()

# Load your dataset (Make sure the path and file name are correct)
@st.cache
def load_data():
    return pd.read_csv('CosmeticsProducts.csv')

data = load_data()

st.title("Cosmetic Product Rating Predictor")

# Display a subset of data
st.header("Sample Data")
st.dataframe(data.sample(10))

# Visualization of average ratings by category
st.header("Average Ratings by Cosmetic Product Category")
chart = alt.Chart(data).mark_bar().encode(
    x=alt.X('category:N', sort='-y'),  # Correct field name used here
    y='average(rating):Q',            # Correct field name used here
    color='category:N',               # Correct field name used here
    tooltip=['category', 'average(rating)']  # Correct field names used here
).properties(
    title='Average Ratings by Cosmetic Product Category',
    width=600
)
st.altair_chart(chart, use_container_width=True)

# User input for ingredients
st.header("Predict Product Rating")
ingredients_input = st.text_area("Enter the ingredients of the cosmetic product:", height=150)

# Predict button
if st.button('Predict Rating') and ingredients_input:
    # Transform the input to the same format as the model was trained on
    transformed_input = tfidf_vectorizer.transform([ingredients_input])
    # Prediction
    prediction = model.predict(transformed_input)
    # Display the prediction
    st.write(f"Predicted Rating: {prediction[0]:.2f}")