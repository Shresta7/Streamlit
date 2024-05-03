import os
import pickle

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

from utils.b2 import B2
from utils.modeling import *

import json
import pickle
import pandas as pd

from utils.b2 import B2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error









# Load configuration variables
with open('./config_vars.json') as f:
    config_vars = json.load(f)

# Initialize the B2 object with configuration
b2 = B2(config_vars['B2_ENDPOINT'],
        config_vars['B2_KEYID'],
        config_vars['B2_APPKEY'])
b2.set_bucket(config_vars['B2_BUCKETNAME'])

# Specify a file path within the bucket to test
file_path = 'CosmeticsProducts.csv'

# Test the bucket connection
B2.test_bucket_connection(b2, file_path)

b2.list_files()
# Load data from B2
data = b2.get_df('CosmeticsProducts.csv')

# print(data.head())
# Prepare data for modeling
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data = data.dropna(subset=['rating', 'ingredients'])

# Text feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limit to 1000 features to avoid overfitting
ingredients_tfidf = tfidf_vectorizer.fit_transform(data['ingredients'])

# Features and target variable
X = ingredients_tfidf
y = data['rating']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
lm = LinearRegression()
lm.fit(X_train, y_train)

# Model evaluation
y_pred = lm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model and the TF-IDF vectorizer
with open('model.pickle', 'wb') as f_model, open('tfidf_vectorizer.pickle', 'wb') as f_tfidf:
    pickle.dump(lm, f_model, pickle.HIGHEST_PROTOCOL)
    pickle.dump(tfidf_vectorizer, f_tfidf, pickle.HIGHEST_PROTOCOL)
