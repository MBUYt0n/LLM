import os
import re
import faiss
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
import kagglehub

# Download dataset
path = kagglehub.dataset_download("shusrith/rag-dataset")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to clean text
def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r'[^a-zA-Z0-9.,!?;:\'"\s]', "", text)
    return text.strip()

# Function to split text into chunks
def split_into_chunks(text, chunk_size=128):
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Load and preprocess PDF data
pdf_path = "combined_document_10.pdf"
raw_text = extract_text_from_pdf(pdf_path)
cleaned_text = clean_text(raw_text)
chunks = split_into_chunks(cleaned_text)

# Encode text chunks
encoding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoding_model.encode(chunks)

# Create FAISS index for text search
d = embeddings.shape[1]
text_index = faiss.IndexFlatL2(d)
text_index.add(embeddings)
index_to_text = {i: chunks[i] for i in range(len(chunks))}

# Load and preprocess tabular data
def table_to_text(row):
    return " | ".join([f"{col}: {str(row[col])}" for col in row.index])

tables = []
table_text = []
row_table_mapping = []

table_files = os.listdir(f"{path}/data/data")
for i, file in enumerate(table_files):
    df = pd.read_csv(f"{path}/data/data/{file}")
    if "Unnamed: 0" in df.columns:
        df.columns = df.columns.str.replace("Unnamed: 0", "Column1")
    tables.append(df)
    table_text.append(df.apply(table_to_text, axis=1))
    for j in range(len(df)):
        row_table_mapping.append((i, j))

# Encode table rows
embd = [encoding_model.encode(t, convert_to_numpy=True) for t in table_text]
table_embeddings_np = np.vstack(embd)

# Create FAISS index for table search
table_index = faiss.IndexFlatL2(384)
table_index.add(table_embeddings_np)
faiss.write_index(table_index, "table_index.faiss")

# Initialize language model
os.environ["GROQ_API_KEY"] = "your_groq_api_key"
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

# Function to generate answer
def generate_answer(query, k=3):
    query_embedding = encoding_model.encode([query]).astype(np.float32)
    
    # Search in FAISS indexes
    distances_text, indices_text = text_index.search(query_embedding, k)
    distances_table, indices_table = table_index.search(query_embedding, k)
    
    # Retrieve matching text chunks
    matched_text = [index_to_text[i] for i in indices_text[0]]
    
    # Retrieve matching table rows
    matched_table = [tables[t].iloc[r] for t, r in [row_table_mapping[i] for i in indices_table[0]]]
    
    # Format context
    context_text = "\n".join(matched_text)
    context_table = "\n".join([str(row) for row in matched_table])
    full_context = f"Textual Data:\n{context_text}\n\nTabular Data:\n{context_table}"
    
    # Generate answer using language model
    prompt = f"""
    You are an AI that answers questions based on provided text and table data.
    
    ### Context:
    {full_context}
    
    ### Question:
    {query}
    
    Provide a concise, well-structured answer.
    """
    messages = [
        SystemMessage(content="You are an AI that answers questions based on provided context."),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content

# Streamlit UI
st.title("End-to-End RAG System Implementation Using LangChain on a 10-page PDF Data Source")
query = st.text_area("Enter your question:")
if query:
    with st.spinner("Generating answer..."):
        answer = generate_answer(query)
    st.subheader("Answer:")
    st.write(answer)
