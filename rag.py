import os
import hashlib
from auto_loader import AutoLoader
from semantic_splitter import SemanticSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever

# Utility to hash the file for unique ID
def hash_file_contents(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Load or create index per doc
def load_or_create_index(file_path: str, index_path: str):
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

    if not os.path.exists(index_path):
        loader = AutoLoader(file_path)
        docs = loader.load()
        if not docs:
            raise ValueError("Failed to load document or document is empty.")

        structured_docs = " ".join([i.page_content.strip() for i in docs])

        splitter = SemanticSplitter(depth='standard').auto_split(structured_docs)
        if not splitter:
            raise ValueError("Semantic splitting failed. No chunks created.")

        db = FAISS.from_documents(splitter, embedding_model)
        db.save_local(index_path)
    else:
        db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Full RAG pipeline
def get_answer(file_path: str, query: str, api_key: str):
    doc_hash = hash_file_contents(file_path)
    index_path = f"indexes/index_{doc_hash}"
    os.makedirs("indexes", exist_ok=True)

    db = load_or_create_index(file_path, index_path)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
    retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.invoke(query)
    return result["result"] if isinstance(result, dict) and "result" in result else str(result)
