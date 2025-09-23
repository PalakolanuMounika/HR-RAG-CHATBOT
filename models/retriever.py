import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and embeddings
def load_index(index_path='embeddings/faiss_index.pkl', emb_path='embeddings/embeddings.npy'):
    index = faiss.read_index(index_path)
    embeddings = np.load(emb_path)
    return index, embeddings

def embed_text(text_list):
    return embedding_model.encode(text_list)

def retrieve(query, chunks, index, top_k=5):
    """Retrieve top-k chunks using FAISS + optional BM25 re-ranking"""
    q_emb = embed_text([query])
    D, I = index.search(q_emb, top_k)  # FAISS search
    top_chunks = [chunks[i] for i in I[0]]

    # Optional BM25 re-ranking
    tokenized = [c.split() for c in top_chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    sorted_chunks = [top_chunks[i] for i in np.argsort(scores)[::-1]]
    
    return sorted_chunks
