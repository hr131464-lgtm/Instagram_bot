from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from load_documents import load_documents
from chunk_documents import create_chunks


def create_vectorstore():
    # Load documents
    documents = load_documents()

    # Split documents into chunks
    chunks = create_chunks(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    # Create embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating embeddings...")

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save vector store
    vectorstore.save_local("vectorstore")

    print("Vector store created successfully.")
    print("Saved to: vectorstore/")


if __name__ == "__main__":
    create_vectorstore()