from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


#Extract text from PDF files
def load_pdf(data):
    loader = DirectoryLoader(
        data, 
        glob="*.pdf", 
        loader_cls=PyPDFLoader
    ) 
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        # Create a new Document with only the page_content and metadata
        src = doc.metadata.get("source")
        minimal_doc = Document(page_content=doc.page_content, metadata={"source": src})
        minimal_docs.append(minimal_doc)
    return minimal_docs


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )

    split_docs = text_splitter.split_documents(minimal_docs)
    return split_docs


def download_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    return embeddings


