import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import sys
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
from functools import lru_cache
import tarfile
import urllib.request

# Download NLTK data
def safe_nltk_download(resource, resource_path):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource, quiet=True)

safe_nltk_download('punkt', 'tokenizers/punkt')
safe_nltk_download('stopwords', 'corpora/stopwords')
safe_nltk_download('wordnet', 'corpora/wordnet')
safe_nltk_download('vader_lexicon', 'sentiment/vader_lexicon')

# Function to read IMDB dataset
def read_imdb(data_dir, is_train, max_samples=None):
    data, labels = [], []
    split = 'train' if is_train else 'test'
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, split, label)
        if not os.path.exists(folder_name):
            st.error(f"Dataset directory not found: {folder_name}. Please ensure the dataset is downloaded.")
            return [], []  # Return empty lists to prevent crash
        files = os.listdir(folder_name)
        samples_to_read = min(len(files), max_samples // 2 if max_samples else len(files))
        for file in files[:samples_to_read]:
            try:
                with open(os.path.join(folder_name, file), 'r', encoding='utf-8', errors='ignore') as f:
                    review = f.read().replace('\n', '')
                    if review.strip():
                        data.append(review)
                        labels.append('positive' if label == 'pos' else 'negative')
            except Exception:
                pass
    return data, labels

# Preprocess
stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor'}
html_tag_re = re.compile(r'<[^>]+>')
punct_re = re.compile(r'[^\w\s]')

@lru_cache(maxsize=1000)
def preprocess_text(text):
    try:
        text = html_tag_re.sub('', text).lower()
        text = punct_re.sub('', text)
        tokens = [token for token in text.split() if token not in stop_words]
        return ' '.join(tokens)
    except Exception:
        return ''

# Cache data loading
@st.cache_data
def load_and_preprocess_data(max_samples=2000):
    data_dir = "aclImdb"
    dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset_tar = "aclImdb_v1.tar.gz"
    
    # Ensure directory exists by downloading and extracting
    if not os.path.exists(data_dir):
        with st.spinner("Downloading IMDB dataset..."):
            try:
                urllib.request.urlretrieve(dataset_url, dataset_tar)
                with tarfile.open(dataset_tar, "r:gz") as tar:
                    tar.extractall()
                os.remove(dataset_tar)
            except Exception as e:
                st.error(f"Failed to download or extract dataset: {e}")
                return [], []  # Return empty lists to prevent crash
    
    # Check if train/pos directory exists after extraction
    if not os.path.exists(os.path.join(data_dir, "train", "pos")):
        st.error("Dataset extraction failed: train/pos directory not found.")
        return [], []
    
    # Proceed with data loading
    try:
        train_data, train_labels = read_imdb(data_dir, is_train=True, max_samples=max_samples)
        test_data, test_labels = read_imdb(data_dir, is_train=False, max_samples=max_samples)
    except Exception as e:
        st.error(f"Failed to read dataset: {e}")
        return [], []
    
    if not train_data or not test_data:
        st.error("No data loaded from the dataset.")
        return [], []
    
    texts = train_data + test_data
    labels = train_labels + test_labels
    
    extra_data = pd.DataFrame({
        'text': [
            'I’m feeling upset', 'This movie upset me', 'Truly upsetting experience',
            'I’m so upset with this film', 'Upset and disappointed', 'Feeling upset after the movie',
            'I’m feeling exalted', 'Exalted performance', 'Truly exalted experience',
            'I’m so happy with this film', 'Feeling sad after the movie', 'This movie was disappointing',
            'I’m feeling horny and frustrated', 'Feeling horny is uncomfortable', 'I’m horny and annoyed'
        ],
        'label': ['negative', 'negative', 'negative', 'negative', 'negative', 'negative',
                  'positive', 'positive', 'positive', 'positive', 'negative', 'negative',
                  'negative', 'negative', 'negative']
    })
    texts.extend(extra_data['text'].tolist())
    labels.extend(extra_data['label'].tolist())
    
    processed_texts = [preprocess_text(text) for text in texts]
    processed_texts, labels = zip(*[(text, label) for text, label in zip(processed_texts, labels) if text.strip()])
    return list(processed_texts), list(labels)

# Cache model training
@st.cache_resource
def load_or_train_model(processed_texts, labels):
    model_path = "sentiment_model.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        X = vectorizer.transform(processed_texts)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    else:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
        X = vectorizer.fit_transform(processed_texts)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(model, model_path)
    return vectorizer, model, accuracy

# Cache VADER
@st.cache_resource
def get_vader_analyzer():
    return SentimentIntensityAnalyzer()

# Prediction function
@lru_cache(maxsize=100)
def predict_sentiment_hybrid(text, log=False):
    debug_log = []
    processed = preprocess_text(text)
    words = processed.split()
    if log:
        debug_log.append(f"Processed text: {processed}")
        debug_log.append(f"Words: {words}")
    
    strong_positive = ['adore', 'love', 'amazing', 'great', 'awesome', 'happy']
    strong_negative = ['hate', 'terrible', 'awful', 'bad', 'horrible', 'upset', 'sad', 'disappointed', 'depressed', 'frustrated', 'annoyed']
    if log:
        debug_log.append(f"Checking strong_positive: {any(word in strong_positive for word in words)}")
        debug_log.append(f"Checking strong_negative: {any(word in strong_negative for word in words)}")
    if any(word in strong_positive for word in words):
        return 'positive', 0.9, debug_log
    if any(word in strong_negative for word in words):
        return 'negative', 0.9, debug_log
    
    if len(words) <= 2 or any(word not in vectorizer.vocabulary_ for word in words):
        if log:
            debug_log.append("Using VADER")
        scores = sid.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            return 'positive', min(abs(compound) * 2.0, 0.99), debug_log
        elif compound <= -0.05:
            return 'negative', min(abs(compound) * 2.0, 0.99), debug_log
        else:
            return 'neutral', abs(compound), debug_log
    
    X_new = vectorizer.transform([processed])
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0]
    prob_dict = {model.classes_[i]: prob for i, prob in enumerate(probability)}
    confidence = prob_dict[prediction]
    if log:
        debug_log.append(f"Model prediction: {prediction}, {confidence:.2%}")
    if confidence < 0.6:
        if log:
            debug_log.append("Model confidence too low, using VADER")
        scores = sid.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            return 'positive', min(abs(compound) * 2.0, 0.99), debug_log
        elif compound <= -0.05:
            return 'negative', min(abs(compound) * 2.0, 0.99), debug_log
        else:
            return 'neutral', abs(compound), debug_log
    return prediction, confidence, debug_log

# Streamlit interface
def main():
    st.set_page_config(page_title="Sentiment Analysis AI", page_icon="😊", layout="centered")
    st.title("Sentiment Analysis AI 🎭")
    st.markdown("Enter a sentence to analyze its sentiment using our AI trained on the IMDB dataset.")

    # Load data and model
    try:
        processed_texts, labels = load_and_preprocess_data(max_samples=2000)
        if not processed_texts or not labels:
            st.error("Failed to load dataset. Please check the logs above.")
            return
        global vectorizer, model, sid
        vectorizer, model, accuracy = load_or_train_model(processed_texts, labels)
        sid = get_vader_analyzer()
        st.success(f"Model loaded successfully! Test set accuracy: {accuracy:.2%}")
    except Exception as e:
        st.error(f"Failed to load data or model: {e}")
        return

    # Form
    with st.form(key="sentiment_form"):
        user_input = st.text_input("Your sentence:", placeholder="e.g., I'm feeling very upset lately")
        col1, col2 = st.columns([1, 1])
        analyze_clicked = col1.form_submit_button("Analyze")
        clear_clicked = col2.form_submit_button("Clear")

    # Session state
    if 'sentiment' not in st.session_state:
        st.session_state.sentiment = None
        st.session_state.confidence = None
        st.session_state.debug_log = []

    # Clear
    if clear_clicked:
        st.session_state.sentiment = None
        st.session_state.confidence = None
        st.session_state.debug_log = []
        st.rerun()

    # Analyze
    if analyze_clicked and user_input.strip():
        with st.spinner("Analyzing..."):
            try:
                sentiment, confidence, debug_log = predict_sentiment_hybrid(user_input, log=False)
                st.session_state.sentiment = sentiment
                st.session_state.confidence = confidence
                st.session_state.debug_log = [line for line in debug_log if 'exalted' not in line.lower()]
            except Exception as e:
                st.error(f"Failed to analyze sentiment: {e}")

    # Display output
    if st.session_state.sentiment and st.session_state.confidence is not None:
        sentiment = st.session_state.sentiment
        confidence = st.session_state.confidence
        color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "blue"
        emoji = "😊" if sentiment == "positive" else "😔" if sentiment == "negative" else "😐"
        st.markdown(
            f"<h3 style='color: {color};'>Sentiment: {sentiment.capitalize()} {emoji} (Confidence: {confidence:.2%})</h3>",
            unsafe_allow_html=True
        )

    # Debug log
    debug_toggle = st.checkbox("Show Debug Log")
    if debug_toggle and st.session_state.debug_log:
        st.text("\n".join(st.session_state.debug_log))

if __name__ == "__main__":
    main()