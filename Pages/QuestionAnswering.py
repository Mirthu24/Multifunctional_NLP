import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

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
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        background-color: #f0fff7;
        border-radius: 10px;
    }
    .stSidebarContent {
        background-color: #75e5b2;
        color: #6d597a;
    }
    </style>
""", unsafe_allow_html=True)


class QuestionAnsweringModel:
    def __init__(self):
        """Initialize and load all necessary models and tokenizers."""
        self.tokenizer_qa = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.model_qa = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    def answer_question(self, question, context):
        """Answer a question based on the given context."""
        inputs = self.tokenizer_qa.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_qa(**inputs)
        
        answer_start, answer_end = torch.argmax(outputs.start_logits), torch.argmax(outputs.end_logits) + 1
        if answer_start >= answer_end:
            return "I couldn't find a relevant answer."
        
        return self.tokenizer_qa.convert_tokens_to_string(
            self.tokenizer_qa.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        ).strip()


# Streamlit UI
st.title("Question Answering System")

# User inputs context paragraph
context = st.text_area("Enter the context paragraph:", height=150)

# User inputs multiple questions (one per line)
questions = st.text_area("Enter one or more questions (each question on a new line):", height=120)

if st.button("Get Answers"):
    if context and questions:
        qa_model = QuestionAnsweringModel()  # Initialize model
        
        question_list = questions.strip().split("\n")  # Split questions by line
        for i, question in enumerate(question_list, 1):
            answer = qa_model.answer_question(question.strip(), context)  # Get answer
            st.success(f"**Q{i}:** {question.strip()}\n\n**Answer:** {answer}")
    else:
        st.warning("Please enter both a context and at least one question.")
