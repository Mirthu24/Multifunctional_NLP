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


class GenerateStory:
    def __init__(self):
        """Initialize and load all necessary models and tokenizers."""
        self.tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
        self.model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    def generate_story(self, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
        """Generate a story based on the given prompt."""
        inputs = self.tokenizer_gpt2(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model_gpt2.generate(
                **inputs, max_length=max_length, temperature=temperature,
                top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                do_sample=True, pad_token_id=self.tokenizer_gpt2.eos_token_id
            )
        return self.tokenizer_gpt2.decode(output[0], skip_special_tokens=True).strip()


# Streamlit UI
st.title("Story Generator")

story_prompt = st.text_input("Enter a story prompt:")

if st.button("Generate Story"):
    if story_prompt:
        generator = GenerateStory()  # Create an instance of GenerateStory
        story = generator.generate_story(story_prompt)  # Generate story
        st.write(story)  # Display generated story
    else:
        st.warning("Please enter a story prompt.")
