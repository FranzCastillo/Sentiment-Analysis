import re
import string

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer


def _preprocess_text(text: str) -> str:
    # Convert text to lowercase
    text = text.lower()

    # Removing links
    text = re.sub(r'http\S+', '', text)

    # Removing mentions and hashtags
    #   Mentions are removed because they are not important,
    #   Only removing the '#' from hashtags because the text in the hashtag might be important
    text = re.sub(r'@\w+|#', '', text)

    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Removing all numbers but 911 because it is an emergency number
    text = re.sub(r'\b(?!911\b)\d+\b', '', text)

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Handle the û symbol
    special_chars = ["\x89û", ""]
    for char in special_chars:
        text = text.replace(char, '')

    return text


class Data:
    def __init__(self, data_path: str, need_download=False):
        if need_download:
            import nltk
            nltk.download('vader_lexicon')
            nltk.download('stopwords')

        self.data = pd.read_csv(data_path)
        self._vectorizer = self._init_vectorizer()
        self.sia = SentimentIntensityAnalyzer()
        self.model = joblib.load('data/model.pkl')

    def _init_vectorizer(self) -> CountVectorizer:
        vectorizer = CountVectorizer()
        vectorizer.fit(self.data['text_clean'])
        return vectorizer

    def _get_sentiment_score(self, text: str) -> float:
        return self.sia.polarity_scores(text)['compound']

    def _get_sentiment_scores(self, texts):
        return np.array([self._get_sentiment_score(text) for text in texts])

    def predict(self, X) -> list:
        # Vectorize the input text
        X_vectorized = self._vectorizer.transform(X)
        sentiment_scores = self._get_sentiment_scores(X)

        # Combine the vectorized text with the sentiment scores
        X_combined = np.hstack([X_vectorized.toarray(), sentiment_scores.reshape(-1, 1)])
        return self.model.predict(X_combined)
