import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import os
from models.document_loader import load_and_clean
from models.chunker import chunk_text, generate_and_save_embeddings
from models.retriever import load_index, retrieve
from models.prompt_templates import rag_prompt
from models.llm_client import ask_llm

st.set_page_config(page_title="HR RAG Chatbot", layout="wide")
st.title("📄 HR RAG Chatbot")

# --- Load and prepare document ---
PDF_FILENAME = "HR-Policy.pdf"
text = load_and_clean(PDF_FILENAME)
chunks = chunk_text(text)

# Embeddings & FAISS index paths
EMB_PATH = 'embeddings/embeddings.npy'
INDEX_PATH = 'embeddings/faiss_index.pkl'

if not os.path.exists(EMB_PATH) or not os.path.exists(INDEX_PATH):
    st.info("Generating embeddings and FAISS index, please wait...")
    generate_and_save_embeddings(chunks, emb_path=EMB_PATH, index_path=INDEX_PATH)

# Load FAISS index
index, embeddings = load_index(index_path=INDEX_PATH, emb_path=EMB_PATH)

# --- User input ---
question = st.text_input("Ask your HR question:")

if question:
    with st.spinner("Fetching answer..."):
        # Retrieve top chunks
        top_chunks = retrieve(question, chunks, index, top_k=5)

        # Build RAG prompt
        prompt = rag_prompt(question, top_chunks)

        # Query LLM
        answer = ask_llm(prompt)

    # Display answer
    st.subheader("Answer:")
    st.write(answer)

    # Optional: show sources
    st.subheader("Source Chunks:")
    for i, chunk in enumerate(top_chunks, 1):
        st.markdown(f"**Chunk {i}:** {chunk[:300]}...")  # show first 300 chars
