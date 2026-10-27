import os
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once per process to avoid repeated startup costs."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_store(
    pdf_path: str,
    vectorstore_path: str
):
    """
    Load a PDF, split it into chunks,
    create embeddings and save them in FAISS.
    """

    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(
        f"\n📄 Loaded {len(documents)} pages"
    )

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"✂️ Created {len(chunks)} chunks"
    )

    # Create embedding model
    embeddings = get_embeddings()

    # Create FAISS vector store
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Create directory if it doesn't exist
    os.makedirs(
        vectorstore_path,
        exist_ok=True
    )

    # Save FAISS index
    vector_store.save_local(
        vectorstore_path
    )

    print(
        f"💾 Vector store saved at: "
        f"{vectorstore_path}"
    )

    return vector_store