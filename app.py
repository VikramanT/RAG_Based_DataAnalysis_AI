import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import requests
import chromadb
import asyncio
from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO

# Fix asyncio event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))  # Initialize a new event loop

# LM Studio API Endpoint
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

# Define paths
DB_PATH = "uploaded_files.db"
UPLOAD_FOLDER = "uploaded_files"
VECTOR_DB_PATH = "chroma_db"

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Initialize SQLite database
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS files (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          filename TEXT,
                          filetype TEXT,
                          filepath TEXT)''')

init_db()

# Save file metadata
def save_file_metadata(filename, filetype, filepath):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO files (filename, filetype, filepath) VALUES (?, ?, ?)", 
                     (filename, filetype, filepath))

# Load stored files metadata
def load_files_from_db():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT id, filename, filetype, filepath FROM files").fetchall()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_or_create_collection(name="rag_collection")

# Initialize embeddings & vector database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

# Function to process and index files into ChromaDB
def process_and_index_file(filepath, filetype):
    try:
        if filetype == "text/csv":
            df = pd.read_csv(filepath)
            text_data = df.to_string()
        elif filetype == "application/json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                text_data = json.dumps(data, indent=2)
        elif filetype == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(filepath)
            text_data = df.to_string()
        else:
            return "Unsupported file type."
        
        # Split text into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text_data)

        # Store document in ChromaDB
        collection.add(
            documents=texts, 
            metadatas=[{"source": filepath} for _ in texts], 
            ids=[f"{filepath}_{i}" for i in range(len(texts))]
        )

        return "File indexed successfully."
    except Exception as e:
        return f"Error processing file: {e}"

# Define retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Function to query LM Studio API
def query_lm_studio(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }

    try:
        response = requests.post(LM_STUDIO_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"API Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Connection error: {e}"

# Streamlit UI
st.title("RAG-Enabled File Query System")
uploaded_file = st.file_uploader("Upload a file (CSV, JSON, Excel)", type=["csv", "json", "xlsx"])

if uploaded_file:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    save_file_metadata(uploaded_file.name, uploaded_file.type, file_path)
    indexing_status = process_and_index_file(file_path, uploaded_file.type)
    st.success(f"File '{uploaded_file.name}' uploaded and indexed! {indexing_status}")

# Display stored files
st.subheader("Stored Files")
stored_files = load_files_from_db()
for file_id, filename, filetype, filepath in stored_files:
    st.write(f"{file_id}: {filename} ({filetype})")

# Query Section
st.subheader("Query the Data")
query = st.text_input("Enter your query:")
if st.button("Search") and query:
    rag_response = retriever.get_relevant_documents(query)  # Retrieve relevant text from indexed files
    context_text = "\n".join([doc.page_content for doc in rag_response])
    
    final_prompt = f"Use the following context to answer:\n\n{context_text}\n\nQuery: {query}"
    response = query_lm_studio(final_prompt)

    st.write("### Response:")
    st.write(response)
