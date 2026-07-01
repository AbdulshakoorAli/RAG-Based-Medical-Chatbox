def build_rag_prompt(context: str) -> str:
    return f"""
You are a medical assistant. Use only the context below to answer the user's question.
If the answer is not found in the provided context, say that you cannot answer from the given sources.

Context:
{context}

"""