import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Helper Functions for File Reading
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def read_excel(file, sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name)
    return df.to_string()

def read_text(file):
    return file.read().decode("utf-8")

# File Handling Function
def load_files(uploaded_files, file_type, sheet_name=None):
    combined_text = ""
    for uploaded_file in uploaded_files:
        if file_type == "Text":
            combined_text += read_text(uploaded_file)
        elif file_type == "PDF":
            combined_text += read_pdf(uploaded_file)
        elif file_type == "CSV":
            combined_text += read_csv(uploaded_file)
        elif file_type == "Excel":
            combined_text += read_excel(uploaded_file, sheet_name)
    return combined_text

# Streamlit UI
st.title("RAG-Based Question Answering System")

# File Upload Section
uploaded_files = st.file_uploader("Upload Files (Text, PDF, CSV, Excel)", type=["txt", "pdf", "csv", "xlsx"],
                                  accept_multiple_files=True)

# File Type Selection
file_type = st.sidebar.selectbox("Select File Type", ["Text", "PDF", "CSV", "Excel"])

# Excel Sheet Selection
sheet_name = None
if file_type == "Excel":
    sheet_name = st.sidebar.text_input("Enter Sheet Name (for Excel files)", "Sheet1")
if file_type == "CSV":
    sheet_name = st.sidebar.text_input("Enter Sheet Name (for CSV files)", "Sheet1")

# Model Selection
model_name = st.sidebar.selectbox(
    "Select LLM Model",
    ["gpt-3.5-turbo (OpenAI)", "gemini-pro (Google)", "all-MiniLM-L6-v2 (HuggingFace)", "llama3-8b-8192 (Groq)", "Custom Local Model"]
)

# API Key Input
api_key = st.sidebar.text_area("Enter API Key")

# Load Data and Split
if uploaded_files and st.button("Process Files"):
    # Load and combine file data
    combined_text = load_files(uploaded_files, file_type, sheet_name)

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.create_documents([combined_text])

    # Embedding Generation
    st.write("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Vector Store Initialization (FAISS)
    st.write("Storing embeddings in FAISS...")
    vector_db = FAISS.from_documents(documents, embedding=embeddings)
    retriever = vector_db.as_retriever()

    # Model Selection
    if "gpt-3.5" in model_name:
        llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")

    elif "gemini-pro" in model_name:
        from langchain.chat_models import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-pro")

    elif "MiniLM" in model_name:
        from transformers import pipeline
        generator = pipeline("text-generation", model="all-MiniLM-L6-v2")
        llm = HuggingFacePipeline(pipeline=generator)

    elif "Groq" in model_name:
        from langchain_groq import ChatGroq
        generator = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")
        llm = load_qa_chain(generator)

    else:
        st.error("Custom local models not implemented yet.")

    # RAG QA Chain
    st.write("Setting up QA system...")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # Question Input and Answer
    question = st.chat_input("Ask a Question:")
    if question:
        response = qa_chain.run(question)
        st.subheader("Answer:")
        st.write(response)
