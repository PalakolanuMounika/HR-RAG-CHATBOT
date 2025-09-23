# testing.py
import os
from models.document_loader import load_and_clean
from models.chunker import chunk_text, generate_and_save_embeddings
from models.retriever import load_index, retrieve
from models.prompt_templates import rag_prompt
from models.llm_client import ask_llm

# 1️⃣ Load and clean PDF
PDF_FILENAME = "HR-Policy.pdf"  # just the file name, in 'data/' folder
text = load_and_clean(PDF_FILENAME)

# 2️⃣ Chunk the text
chunks = chunk_text(text)

# 3️⃣ Generate embeddings & FAISS index if not already done
EMB_PATH = 'embeddings/embeddings.npy'
INDEX_PATH = 'embeddings/faiss_index.pkl'

if not os.path.exists(EMB_PATH) or not os.path.exists(INDEX_PATH):
    print("Generating embeddings and FAISS index...")
    generate_and_save_embeddings(chunks, emb_path=EMB_PATH, index_path=INDEX_PATH)
else:
    print("Embeddings and FAISS index already exist. Loading...")

# 4️⃣ Load FAISS index
index, embeddings = load_index(index_path=INDEX_PATH, emb_path=EMB_PATH)

# 5️⃣ Ask a question
question = input("Ask your HR question: ")

# 6️⃣ Retrieve relevant chunks
top_chunks = retrieve(question, chunks, index, top_k=5)

# 7️⃣ Build RAG prompt
prompt = rag_prompt(question, top_chunks)

# 8️⃣ Query LLM
answer = ask_llm(prompt)

# 9️⃣ Print the answer
print("\n✅ Answer:", answer)
