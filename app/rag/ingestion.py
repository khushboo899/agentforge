import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def create_vector_store(pdf_path: str, vectorstore_path: str):

    print("\n📄 Loading PDF...")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(f"Loaded {len(documents)} pages")


    print("\n✂️ Splitting document into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")


    print("\n🧮 Creating embeddings...")

    embeddings = get_embeddings()


    print("\n🗄️ Creating FAISS vector store...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )


    os.makedirs(vectorstore_path, exist_ok=True)

    vector_store.save_local(vectorstore_path)

    print("✅ Vector store created and saved")

    return vector_store