import string
import re
import pandas as pd
import numpy as np
import translators as ts
import nltk

nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class TextPreprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.stemmer = SnowballStemmer("english")

    def preprocess_text(text):
        """
        Preprocesses text by converting to lowercase, removing punctuation, and tokenizing.
        """
        lower_text = text.lower()   # Convert text to lower
        print("\nCase Folding : ", lower_text)

        token_text = re.sub(r"\d+", "", lower_text)    # Remove number
        token_text = ''.join(c for c in token_text if c not in string.punctuation)    # Remove punctuation
        text_tokens = word_tokenize(token_text)    # Tokenize the text
        freq_tokens = nltk.FreqDist(text_tokens)    # Frequency word token
        print("\nRemove punctuation, number, multiple whitespace : ", token_text)
        print("\nTokenizing Result : ", text_tokens)
        print("\nFrequency Result : ", freq_tokens.most_common())

        stop_words = [word for word in freq_tokens if word not in stopwords.words('english')]   # Implement stopwords
        
        stemmer = SnowballStemmer('english')  # Initialized stemmer
        stemmed_words = [stemmer.stem(word) for word in stop_words]   # Stem the words

        normalized_text = ''.join(stemmed_words)  # Join the words
        
        return normalized_text
        # raise NotImplementedError

    def preprocess_data(df, lang):
        """
        Preprocesses the text in the 'review' column of the given DataFrame.
        """
        df['review'] = df['review'].apply(lambda x: TextPreprocessor.preprocess_text(x, lang))
        
        return df
        # raise NotImplementedError

def tfidf_vectorizer(text, df):
    """
    Converts the 'review' column of the given DataFrame into its TF-IDF vectorized form.
    """
    # Preprocess the text
    df = TextPreprocessor.preprocess_data(df, 'english')

    # Create Tfidf vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Convert the 'review' column into its TF-IDF vectorized form
    X = vectorizer.fit_transform(df['review'])

    return X, vectorizer
    # raise NotImplementedError

def train_and_evaluate_data(df, data_path):
    """
    Trains a Naive Bayes classifier on the given data and evaluates its performance using cross validation and
    sum data train using confusion matrix
    Parameters:
    - df (DataFrame) : The preprocessed dataset
    - data_path (str) : The path to the dataset file
    Returns:
    - None
    """
    # Create training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    # Convert training and test datasets into vectorized form
    X_train_vectorized, vectorizer = tfidf_vectorizer(df)
    X_test_vectorized, = vectorizer.transform(X_test)

    # Create a Multinomial Naive Bayes model
    model = MultinomialNB(alpha=1)

    # Train the model
    model.fit(X_train_vectorized, y_train)

    # Predict sentiment for test dataset
    y_pred = model.predict(X_test_vectorized)

    # Calculate accuracy of the model
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print(f'Accuracy of the model: {accuracy*100:.2f}%')
    # raise NotImplementedError
