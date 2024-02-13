import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import re
import string

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function for text preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Spelling correction, remove special characters, digits, and perform tokenization
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9]", '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('@\w+', '', text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove stopwords, perform stemming, and lemmatization
    tokens = [ps.stem(lemmatizer.lemmatize(word.lower())) for word in tokens if word.isalnum() and word.lower() not in stop_words]

    return ' '.join(tokens)

# Load the dataset
df = pd.read_csv('financial_sentiment_data.csv')

# Perform Vader Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
df['Vader_Sentiment'] = df['Sentence'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Encode Sentiment labels
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Streamlit app
st.title('Sentiment Analysis App')

# User input
user_input = st.text_input('Enter a sentence:')
user_input_clean = preprocess_text(user_input)


if 'pipeline' not in st.session_state:
    # Build the pipeline with CountVectorizer and AdaBoostClassifier
    st.session_state.pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', AdaBoostClassifier(
            n_estimators=800, 
            learning_rate=0.5))
    ])

    # Train the model only once
    st.session_state.pipeline.fit(df['Sentence'], df['Sentiment'])

# Predict sentiment using the trained model
if st.button('Predict'):
    user_prediction = st.session_state.pipeline.predict([user_input_clean])[0]
    user_prediction_label = label_encoder.inverse_transform([user_prediction])[0].upper()

# Determine color and emoji based on sentiment
    color = 'green' if user_prediction_label == 'POSITIVE' else ('yellow' if user_prediction_label == 'NEUTRAL' else 'red')
    emoji = '😊' if user_prediction_label == 'POSITIVE' else ('😐' if user_prediction_label == 'NEUTRAL' else '😔')

    # Display results with colored box and big emoji
    st.markdown(f'<div style="padding: 10px; color: white; background-color: {color}; border-radius: 5px;">\
                <h3>Cleaned Sentence:</h3>\
                <p>{user_input_clean}</p>\
                <h3>Predicted Sentiment:</h3>\
                <p style="font-size: 48px;"><span>{emoji}</span> {user_prediction_label}</p>\
                </div>', unsafe_allow_html=True)





