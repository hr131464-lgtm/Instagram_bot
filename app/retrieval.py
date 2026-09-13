from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = Path("vectorstore")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3
MAX_DISTANCE = 0.90


@lru_cache(maxsize=1)
def load_vectorstore():
    if not VECTORSTORE_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found. Run "
            "'python app/create_vectorstore.py' first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def retrieve_documents(question: str, top_k: int = TOP_K):
    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        question,
        k=top_k
    )

    retrieved = []

    for document, score in results:
        retrieved.append({
            "content": document.page_content,
            "source": document.metadata.get(
                "source",
                "unknown"
            ),
            "distance": float(score)
        })

    return retrieved


def calculate_retrieval_confidence(retrieved_documents):
    if not retrieved_documents:
        return 0.0

    best_distance = retrieved_documents[0]["distance"]

    confidence = max(
        0.0,
        min(
            1.0,
            1 - (best_distance / MAX_DISTANCE)
        )
    )

    return round(confidence, 3)


def is_relevant(retrieved_documents):
    if not retrieved_documents:
        return False

    best_distance = retrieved_documents[0]["distance"]

    return best_distance <= MAX_DISTANCE