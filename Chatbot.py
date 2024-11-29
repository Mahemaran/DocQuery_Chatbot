import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_name = "google/flan-t5-xxl"  # You can replace this with a smaller model if needed
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Streamlit App
st.title("Dark Psychology Explanation")
st.write("Enter a query, and I will explain dark psychology concepts.")

# Text input for user query
input_text = st.text_input("Your Question", "explain dark psychology?")

if input_text:
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate response
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display response
    st.write("### Response:")
    st.write(response)
