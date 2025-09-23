def chunk_text(text: str, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def generate_and_save_embeddings(chunks, emb_path='embeddings/embeddings.npy', index_path='embeddings/faiss_index.pkl'):
    from models.retriever import embed_text
    import faiss
    import numpy as np
    import os

    embeddings = embed_text(chunks)
    np.save(emb_path, embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    print(f"✅ Saved embeddings to {emb_path} and FAISS index to {index_path}")
