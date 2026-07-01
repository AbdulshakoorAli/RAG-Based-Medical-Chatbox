from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeVectorStore
from src.helper import filter_to_minimal_docs, load_pdf, text_split, download_embeddings
from src.prompt import build_rag_prompt
from src.ssl_utils import configure_ssl_environment
configure_ssl_environment()
from ollama import chat

os.chdir("D:/RAG-Based-Medical-Chatbox")
index_name = "medical-chatbox"
load_dotenv()
extracted_data = load_pdf("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_split_docs = text_split(minimal_docs)
embeddings = download_embeddings()



PINE_CONE_API_KEY = os.getenv("PINE_CONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINE_CONE_API_KEY
pc = Pinecone(api_key = PINE_CONE_API_KEY)

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_split_docs,
    embedding=embeddings,
    index_name = index_name
)


# Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

retriever = docsearch.as_retriever(search_type = "similarity" , search_kwargs={"k":3})

question = "What is treatment of Acne?"

# Retrieve relevant chunks from the vector store.
retrieved_docs = retriever.invoke(question)

context = "\n\n".join(
    f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content.strip()}"
    for doc in retrieved_docs
)

response = chat(
    model="Qwen2.5-Coder",
    messages=[
        {"role": "system", "content": build_rag_prompt(context)},
        {"role": "user", "content": question}
    ]
)

answer = response["message"]["content"]
print(answer)
