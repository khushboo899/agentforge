import os
import hashlib

from langchain_community.vectorstores import FAISS

from app.rag.ingestion import (
    create_vector_store,
    get_embeddings
)


VECTORSTORE_BASE_PATH = "vectorstore"
SIMILARITY_THRESHOLD=1.0


def get_document_id(pdf_path: str):
    """
    Create a unique ID for each document.
    """

    file_name = os.path.basename(pdf_path)

    file_hash = hashlib.md5(
        pdf_path.encode()
    ).hexdigest()[:8]

    document_id = f"{file_name}_{file_hash}"

    return document_id


def get_vector_store_path(pdf_path: str):
    """
    Create a separate vectorstore path
    for each PDF.
    """

    document_id = get_document_id(pdf_path)

    vectorstore_path = os.path.join(
        VECTORSTORE_BASE_PATH,
        document_id
    )

    return vectorstore_path


def get_vector_store(pdf_path: str):

    embeddings = get_embeddings()

    vectorstore_path = get_vector_store_path(
        pdf_path
    )

    index_file = os.path.join(
        vectorstore_path,
        "index.faiss"
    )

    # Existing vector store
    if os.path.exists(index_file):

        print(
            "\n⚡ Loading existing vector store..."
        )

        vector_store = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # New document
    else:

        print(
            "\n🏗️ Creating new vector store..."
        )

        vector_store = create_vector_store(
            pdf_path=pdf_path,
            vectorstore_path=vectorstore_path
        )

    return vector_store


def retrieve_documents(
    question: str,
    pdf_path: str
):

    vector_store = get_vector_store(pdf_path)

    print("\n🔍 Searching relevant document chunks...")

    documents_with_scores = (
        vector_store.similarity_search_with_score(
            question,
            k=4
        )
    )

    results = []

    for i, (document, score) in enumerate(
        documents_with_scores,
        start=1
    ):

        print(
            f"\n📊 Chunk {i} similarity score: {score:.4f}"
        )

        # Reject irrelevant chunks
        if score > SIMILARITY_THRESHOLD:

            print("❌ Rejected: Low relevance")

            continue

        print("✅ Accepted: Relevant chunk")

        page_number = document.metadata.get(
            "page",
            "Unknown"
        )

        source_file = document.metadata.get(
            "source",
            "Unknown"
        )

        result = {
            "content": document.page_content,
            "page": page_number,
            "source": source_file,
            "score": float(score)
        }

        results.append(result)

        print(f"📄 Page: {page_number}")
        print(f"📁 Source: {source_file}")
        print(document.page_content[:300])
        print("-" * 50)

    if not results:

        print(
            "\n⚠️ No sufficiently relevant chunks found."
        )

    return results

    # Everything below must be INSIDE retrieve_documents()
   


    print(f"\n📌 Retrieved chunk {i}")
    print(f"📄 Page: {page_number}")
    print(f"📁 Source: {source_file}")
    print(document.page_content[:300])
    print("-" * 50)

    return results