from process import *

import sys
from textblob import TextBlob

def print_text(text):
    text = TextPreprocessor.preprocess_text(text)
    # Print the original and preprocessed texts
    print("Preprocess Text : ", text)
    
    # Perform sentiment analysis
    analysis = TextBlob(text)

    # Get sentiment polarity (-1 to 1: negative to positive)
    sentiment = analysis.sentiment.polarity
    print(f"Sentiment Polarity : {sentiment}")
    if sentiment > 0: 
        print("Positive sentiment")
    elif sentiment < 0: 
        print("Negative sentiment")
    else: 
        print("Neutral sentiment")
    # raise NotImplementedError

def visual():
    """
    for visual dataset
    """
    raise NotImplementedError

def load_data(df, data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Preprocess the data
    preprocessor = TextPreprocessor()
    preprocessor.preprocess_data(df, 'english')
    # Preprocess and vectorize the data
    X_train_vectorized, vectorizer = tfidf_vectorizer(df)

    # Train and evaluate the model
    train_and_evaluate_data(df, data_path)



def main():
    if len(sys.argv) > 2:
        sys.exit("Usage : python sentiment.py <data.csv> or sentiment.py")
    elif len(sys.argv) == 2:
        data_path = sys.argv[1] 
        print(train_and_evaluate_data(data_path))
    else:
        text = text(input("Text : "))
        print_text(text)


if __name__ == "__main__":
    main()