import os
import shutil
import streamlit as st
from rag import show_chat, chat_input_handler
from collections import deque

# --- 🔧 Directories setup ---
UPLOAD_DIR = "uploaded_docs"
INDEX_DIR = "indexes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

st.title("💬 Your intelligent assistant for document processing, Q&A")

# --- 🔧 Upload and Input
uploaded_file = st.file_uploader("📎 Upload a document", type=["pdf", "txt", "xlsx", "csv", "pptx", "eml"])
query = st.chat_input("💡 Ask your question")

# --- 🔧 API Key
with st.sidebar:
    api_key = st.text_input("🔑 Enter your Gemini API Key", type="password")
    if not api_key:
        st.error("Please type the API KEY")

# --- 🔧 Session State Initialization
if "show_delete_button" not in st.session_state:
    st.session_state.show_delete_button = False

if "messages" not in st.session_state:
    st.session_state["messages"] = deque(maxlen=6)  # ✅ Keep last 3 rounds

# --- 🔧 Chat Processing
if uploaded_file and query and api_key:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        show_chat(st.session_state["messages"])  # ✅ Show previous
        chat_input_handler(query, api_key, file_path)  # ✅ Handle current
        st.session_state.show_delete_button = True

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# --- 🔧 Sidebar Cache Deletion
with st.sidebar:
    if st.session_state.show_delete_button:
        if st.button("🗑️ Delete All Cache") and uploaded_file:
            try:
                with st.spinner("Deleting all files and indexes..."):
                    if os.path.exists(UPLOAD_DIR):
                        shutil.rmtree(UPLOAD_DIR)
                        os.makedirs(UPLOAD_DIR)
                    if os.path.exists(INDEX_DIR):
                        shutil.rmtree(INDEX_DIR)
                        os.makedirs(INDEX_DIR)

                    st.session_state["messages"].clear()  # ✅ Clear messages
                    st.success("All uploaded files and indexes have been deleted.")
            except Exception as e:
                st.error(f"❌ Error while deleting: {str(e)}")
