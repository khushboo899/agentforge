import os
import hashlib

from langchain_community.vectorstores import FAISS

from app.rag.ingestion import (
    create_vector_store,
    get_embeddings
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VECTORSTORE_BASE_PATH = os.path.join(PROJECT_ROOT, "vectorstore")

# Number of candidates retrieved from FAISS
RETRIEVAL_K = 8

# Final number of chunks returned
FINAL_K = 5


# ============================================================
# DOCUMENT ID
# ============================================================

def get_document_id(pdf_path: str):
    """
    Create a stable ID for a PDF.

    The ID is based on:
        - filename
        - path hash

    This allows every PDF to have its own vector store.
    """

    file_name = os.path.basename(pdf_path)
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as pdf_file:
        for chunk in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(chunk)

    # Content-based identity invalidates the index when the PDF changes.
    return f"{file_name}_{digest.hexdigest()[:12]}"


# ============================================================
# VECTOR STORE PATH
# ============================================================

def get_vector_store_path(pdf_path: str):
    """
    Return the directory where the PDF's FAISS index is stored.
    """

    document_id = get_document_id(
        pdf_path
    )

    return os.path.join(
        VECTORSTORE_BASE_PATH,
        document_id
    )


# ============================================================
# LOAD OR CREATE VECTOR STORE
# ============================================================

def get_vector_store(pdf_path: str):
    """
    Load an existing FAISS index.

    If it doesn't exist, create it from the PDF.
    """

    embeddings = get_embeddings()

    vectorstore_path = get_vector_store_path(
        pdf_path
    )

    index_file = os.path.join(
        vectorstore_path,
        "index.faiss"
    )

    # --------------------------------------------------------
    # Existing vector store
    # --------------------------------------------------------

    if os.path.exists(index_file):

        print(
            "\n⚡ Loading existing vector store..."
        )

        vector_store = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # --------------------------------------------------------
    # New vector store
    # --------------------------------------------------------

    else:

        print(
            "\n🏗️ Creating new vector store..."
        )

        vector_store = create_vector_store(
            pdf_path=pdf_path,
            vectorstore_path=vectorstore_path
        )

    return vector_store


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def semantic_search(
    vector_store,
    query: str,
    k: int = RETRIEVAL_K
):
    """
    Perform semantic vector search.

    The embedding model converts the query into a vector.
    FAISS then finds semantically similar document chunks.

    Lower FAISS distance generally means closer vectors.
    """

    results = (
        vector_store
        .similarity_search_with_score(
            query,
            k=k
        )
    )

    return results


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question: str,
    pdf_path: str
):
    """
    Retrieve relevant chunks from a PDF.

    This function intentionally returns more candidates
    than the final answer needs.

    Why?

        Retrieval  → broad candidate collection
        Reranking  → stronger relevance selection

    This is more robust than immediately throwing away
    everything above one fixed FAISS threshold.
    """

    vector_store = get_vector_store(
        pdf_path
    )

    print(
        "\n🔍 Searching relevant document chunks..."
    )

    documents_with_scores = semantic_search(
        vector_store=vector_store,
        query=question,
        k=RETRIEVAL_K
    )

    results = []

    for index, (
        document,
        distance
    ) in enumerate(
        documents_with_scores,
        start=1
    ):

        print(
            f"\n📊 Candidate {index}"
        )

        print(
            f"Distance: {distance:.4f}"
        )

        page_number = document.metadata.get("page", "Unknown")
        # PyPDFLoader stores zero-based page indices; display human page numbers.
        if isinstance(page_number, int):
            page_number += 1

        source_file = document.metadata.get(
            "source",
            "Unknown"
        )

        content = document.page_content

        result = {
            "content": content,
            "page": page_number,
            "source": source_file,
            "score": float(distance),
            "retrieval_method": "semantic"
        }

        results.append(
            result
        )

        print(
            f"📄 Page: {page_number}"
        )

        print(
            f"📁 Source: {source_file}"
        )

        print(
            content[:250]
        )

        print(
            "-" * 50
        )

    if not results:

        print(
            "\n⚠️ No candidates retrieved."
        )

    return results