def rag_prompt(question: str, context_chunks: list) -> str:
    """
    Build RAG prompt combining retrieved context and user question
    """
    context_text = "\n".join(context_chunks)
    prompt = f"""
You are an HR assistant. Use the following HR document excerpts to answer the question.
If the answer is not in the documents, say "I don't know".

Context:
{context_text}

Question: {question}
Answer:
"""
    return prompt


