import re

import pandas as pd
import plotly.express as px

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def prepare_data(df, text_col, target_col):
    """
    Prepares the data for modeling by dropping missing values and converting text to lower case.
    """
    df = df.dropna(subset=[text_col, target_col])
    df[text_col] = df[text_col].str.lower()
    return df

def get_tfidf_features(df, text_col):
    """
    Converts text data into TF-IDF features.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
    return tfidf_matrix, vectorizer

def get_sentiment_data(df, text_col, analyzer):
    """
    Computes sentiment scores for each text entry in the dataframe.
    """
    df_sentiment = []

    for text in df[text_col]:
        scores = analyzer.polarity_scores(text)
        df_sentiment.append(scores)

    df_sentiment = pd.DataFrame(df_sentiment, index=df.index)
    df_sentiment = pd.concat([df, df_sentiment], axis=1)
    return df_sentiment

def get_sentence_sentiment(text, analyzer):
    """
    Breaks text into sentences and computes sentiment for each sentence.
    """
    sentences = re.split('[?.!]', text)
    sentences = [s.strip() for s in sentences if s]
    df_sentences = pd.DataFrame(sentences, columns=['text'])
    return get_sentiment_data(df_sentences, 'text', analyzer)
