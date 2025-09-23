# HR RAG Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** for HR documents built with **Python**, **FAISS**, and **Streamlit**, using **Groq/OpenRouter LLM**. This chatbot can answer HR-related questions by retrieving relevant information from your PDF policies.

---

## Project Structure

# RAG HR Chatbot Project Structure

rag_hr_chatbot/
│
├─ app/
│  └─ main.py                  # Streamlit chat interface
│
├─ data/
│  └─ HR-Policy.pdf            # HR PDF documents
│
├─ embeddings/
│  ├─ embeddings.npy           # Generated embeddings
│  └─ faiss_index.pkl          # FAISS vector index
│
├─ models/
│  ├─ __init__.py
│  ├─ document_loader.py       # Load & clean PDFs
│  ├─ chunker.py               # Chunk text & generate embeddings
│  ├─ retriever.py             # FAISS + BM25 retrieval
│  ├─ prompt_templates.py      # RAG prompt templates
│  └─ llm_client.py            # Groq/OpenRouter LLM wrapper
│
├─ virtual_env/                # Python virtual environment
├─ requirements.txt            # Dependencies
├─ .env                        # API keys (OPENROUTER_API_KEY)
├─ Dockerfile                  # For containerization (optional)
└─ README.md                   # Project documentation


## Setup Instructions

1. **Clone the repository**

```bash
git clone <your-github-repo-url>
cd rag_hr_chatbot

```

2. **Create and activate virtual environment**
```bash

python -m venv virtual_env
virtual_env\Scripts\activate       # Windows
# source virtual_env/bin/activate  # Linux / Mac
```


3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add API key**

Create a .env file in the project root:
```bash

OPENROUTER_API_KEY=<your_openrouter_api_key>
```

## Running the Chatbot

```bash
streamlit run app/main.py
```

- A browser window will open with the HR chatbot interface.

- Type your HR-related questions (e.g., "What is the leave policy?") and get answers based on your PDF documents.

# Usage Workflow

1. **Load HR PDFs**  
   Place your HR PDF documents in the `data/` folder.

2. **Clean & Chunk Text**  
   Split large documents into smaller, manageable segments.

3. **Generate Embeddings**  
   Create embeddings for each text chunk and store them in `embeddings/`.

4. **Build FAISS Index**  
   Enable efficient retrieval of relevant chunks using FAISS.

5. **Retrieve & Rank**  
   Use FAISS with optional BM25 scoring to find and rank relevant chunks.

6. **Build RAG Prompt**  
   Combine the retrieved chunks with the user query to form the RAG prompt.

7. **Query LLM**  
   Use Groq/OpenRouter LLM to generate answers based on the RAG prompt.

8. **Display in Streamlit UI**  
   Show the answer along with the source chunks in the user interface.

---

# Optional Improvements

- Add caching to avoid repeated LLM calls for the same question.  
- Display source pages or chunks in the UI.  
- Containerize using Docker for easier deployment.

---

# Dependencies

- Python 3.10+  
- Streamlit  
- FAISS  
- Sentence Transformers  
- OpenRouter Python SDK  
- PyPDF2  
- Rank-BM25  
- python-dotenv  

Install all dependencies via:

```bash
pip install -r requirements.txt
``