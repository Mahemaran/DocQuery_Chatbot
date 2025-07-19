import os
import shutil
import streamlit as st
from rag import get_answer, hash_file_contents

# Set up directories
UPLOAD_DIR = "uploaded_docs"
INDEX_DIR = "indexes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

st.title("Your intelligent assistant for document processing, Q&A")

uploaded_file = st.file_uploader("Upload a document")
query = st.text_input("Enter your question")
with st.sidebar:
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    if not api_key:
        st.error("Please type the API KEY")

if "show_delete_button" not in st.session_state:
    st.session_state.show_delete_button = False

if st.button("Answer") and uploaded_file and query and api_key:
    # Save file to UPLOAD_DIR
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        with st.spinner("Processing..."):
            answer = get_answer(file_path=file_path, query=query, api_key=api_key)
            st.success(answer)
            st.session_state.show_delete_button = True
    except Exception as e:
        st.error(f"Error: {str(e)}")

with st.sidebar:
    if st.session_state.show_delete_button:
        if st.button("Delete All Cache") and uploaded_file:
            try:
                with st.spinner("Deleting all files and indexes..."):
                    # Delete uploaded documents
                    if os.path.exists(UPLOAD_DIR):
                        shutil.rmtree(UPLOAD_DIR)
                        os.makedirs(UPLOAD_DIR)

                    # Delete FAISS indexes
                    if os.path.exists(INDEX_DIR):
                        shutil.rmtree(INDEX_DIR)
                        os.makedirs(INDEX_DIR)

                    st.success("All uploaded files and indexes have been deleted.")
            except Exception as e:
                st.error(f"Error while deleting: {str(e)}")
