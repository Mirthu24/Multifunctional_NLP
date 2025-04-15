import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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


class NextWordPrediction:
    def __init__(self):
        """Initialize and load all necessary models and tokenizers."""
        self.tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
        self.model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    
    def predict_next_word(self, prompt, top_k=5):
        """Predict the next word based on the given prompt."""
        inputs = self.tokenizer_gpt2(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model_gpt2(**inputs)
        top_k_tokens = torch.topk(outputs.logits[:, -1, :], top_k).indices[0].tolist()
        return [self.tokenizer_gpt2.decode([token]).strip() for token in top_k_tokens]


# Streamlit UI
st.title("Next Word Prediction Tool")

text = st.text_area("Enter text to predict the next word:")

if st.button("Predict"):
    if text:
        predictor = NextWordPrediction()  # Create an instance of NextWordPrediction
        predictions = predictor.predict_next_word(text)  # Get predictions
        st.info(", ".join(predictions))  # Display predictions
    else:
        st.warning("Please enter a prompt.")
