import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Apply Custom CSS for Pastel Theme
st.markdown("""
    <style>
    body {
        background-color: #379371;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #379371;
    }
    .stButton>button {
        background-color: #eafcbd;
        color: #379371;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        background-color: #f0fff7;
        border-radius: 10px;
    }
    .stSidebarContent {
        background-color: #75e5b2;
        color: #6d597a;
    }
    </style>
""", unsafe_allow_html=True)


class SentimentAnalyzer:
    def __init__(self):
        """Initialize and load all necessary models and tokenizers."""
        self.tokenizer_sent = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model_sent = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def analyze_sentiment(self, text):
        """Analyze sentiment of the given text."""
        inputs = self.tokenizer_sent(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model_sent(**inputs)
        return "Positive" if torch.argmax(outputs.logits, dim=1).item() == 1 else "Negative"


# Streamlit UI
st.title("Sentiment Analysis")

sentiment_text = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if sentiment_text:
        analyzer = SentimentAnalyzer()  # Create an instance of SentimentAnalyzer
        sentiment = analyzer.analyze_sentiment(sentiment_text)  # Perform sentiment analysis
        st.success(f"Sentiment: {sentiment}")  # Display sentiment result
    else:
        st.warning("Please enter text.")
