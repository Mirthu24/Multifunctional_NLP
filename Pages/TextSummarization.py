import streamlit as st
from transformers import (BartForConditionalGeneration, BartTokenizer)

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

class TextSummarization:
    def __init__(self):
        """Initialize and load all necessary models and tokenizers."""
        self.tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    def summarize_text(self, text, max_length=130, min_length=30):
        """Generate a summary for the input text."""
        inputs = self.tokenizer_bart.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model_bart.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit UI
st.title("Text Summarization Tool")

text = st.text_area("Enter text to summarize:")
if st.button("Summarize"):
    if text:
        summarizer = TextSummarization()  # Create an instance of the class
        summary = summarizer.summarize_text(text)  # Call the method properly
        st.success(summary)
    else:
        st.warning("Please enter some text.")
