from pathlib import Path


def load_documents():
    knowledge_base_path = Path("knowledge_base")

    documents = []

    for file_path in knowledge_base_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "content": text
        })

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Loaded {len(documents)} document(s).")

    for document in documents:
        print(f"\nSource: {document['source']}")
        print(document["content"][:500])