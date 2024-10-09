from flask import Flask, render_template, request
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('Dataset-SA.csv')

# Predict sentiment using VADER
df['Predicted_Sentiment'] = df['Review'].apply(lambda x: 'neutral' if pd.isna(x) 
                                               else 'positive' if sia.polarity_scores(x)['compound'] > 0 
                                               else 'negative' if sia.polarity_scores(x)['compound'] < 0 
                                               else 'neutral')

# Preprocess text data for BM25 and TF-IDF
df['Review'] = df['Review'].fillna('')

# Tokenize reviews for BM25
tokenized_corpus = [review.split() for review in df['Review']]
bm25_model = BM25Okapi(tokenized_corpus)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Review'])

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST' and 'keyword' in request.form:
        keyword = request.form['keyword'].lower()
        sentiment_filter = request.form['sentiment']
        min_rating = request.form.get('min_rating', type=float, default=0.0)
        max_price = request.form.get('max_price', type=float, default=float('inf'))

        # Step 1: Perform TF-IDF search
        query_vector = tfidf_vectorizer.transform([keyword])
        cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Step 2: Apply BM25 ranking
        bm25_scores = bm25_model.get_scores(keyword.split())

        # Combine both TF-IDF and BM25 rankings
        combined_scores = 0.5 * cosine_sim + 0.5 * bm25_scores

        # Filter based on rating, price, and sentiment
        filtered_df = df.copy()
        filtered_df['score'] = combined_scores
        filtered_df = filtered_df[filtered_df['score'] > 0]
        
        # Convert 'Rate' column to numeric and handle non-numeric values
        filtered_df['Rate'] = pd.to_numeric(filtered_df['Rate'], errors='coerce')
        filtered_df = filtered_df[filtered_df['Rate'] >= min_rating]
        
        # Convert 'product_price' to numeric and handle non-numeric values
        filtered_df['product_price'] = pd.to_numeric(filtered_df['product_price'], errors='coerce')
        filtered_df = filtered_df[filtered_df['product_price'] <= max_price]
        
        if sentiment_filter:
            filtered_df = filtered_df[filtered_df['Predicted_Sentiment'] == sentiment_filter]

        # Sort by score and return relevant results
        filtered_df = filtered_df.sort_values(by='score', ascending=False)
        results = filtered_df[['product_name', 'product_price', 'Rate', 'Review', 'Predicted_Sentiment']]

    return render_template('index.html', tables=[results.to_html(classes='data')] if results is not None else None)

@app.route('/predict', methods=['GET', 'POST'])
def predict_sentiment():
    predicted_sentiment = None
    if request.method == 'POST' and 'user_input' in request.form:
        user_input = request.form['user_input']
        if user_input:
            score = sia.polarity_scores(user_input)['compound']
            if score > 0:
                predicted_sentiment = 'positive'
            elif score < 0:
                predicted_sentiment = 'negative'
            else:
                predicted_sentiment = 'neutral'

    return render_template('predict.html', sentiment=predicted_sentiment)

if __name__ == '__main__':
    app.run(debug=True)
