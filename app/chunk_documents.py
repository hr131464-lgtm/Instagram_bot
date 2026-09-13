from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_documents import load_documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []

    for document in documents:
        document_chunks = text_splitter.create_documents(
            [document["content"]],
            metadatas=[{"source": document["source"]}]
        )

        chunks.extend(document_chunks)

    return chunks


if __name__ == "__main__":
    documents = load_documents()

    chunks = create_chunks(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content)
        print(f"Source: {chunk.metadata['source']}")