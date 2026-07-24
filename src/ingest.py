import os
import glob
import chromadb

KNOWLEDGE_BASE_DIR = os.path.join("data", "knowledge_base")
CHROMA_DB_DIR = os.path.join("data", "chroma_db")
COLLECTION_NAME = "it_knowledge_base"


def load_documents(folder: str) -> list[dict]:
    """Read every .md file in the folder. Returns a list of {filename, text}."""
    documents = []
    filepaths = glob.glob(os.path.join(folder, "*.md"))

    if not filepaths:
        raise FileNotFoundError(
            f"No .md files found in {folder}. Did you add the knowledge base docs?"
        )

    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        filename = os.path.basename(path)
        documents.append({"filename": filename, "text": text})

    return documents


def chunk_by_paragraph(text: str) -> list[str]:
    raw_chunks = text.split("\n\n")
    chunks = []
    for chunk in raw_chunks:
        cleaned = chunk.strip()
        if len(cleaned) > 20:
            chunks.append(cleaned)
    return chunks


def build_index():
    print(f"Loading documents from {KNOWLEDGE_BASE_DIR} ...")
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    print(f"Found {len(documents)} documents.\n")

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Wipe and recreate the collection each time we re-ingest
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet, that's fine

    collection = client.create_collection(name=COLLECTION_NAME)

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for doc in documents:
        chunks = chunk_by_paragraph(doc["text"])
        print(f"  {doc['filename']}: {len(chunks)} chunks")

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc['filename']}_chunk_{i}"
            all_chunks.append(chunk_text)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source_document": doc["filename"],
                "chunk_index": i,
            })

    print(f"\nTotal chunks to store: {len(all_chunks)}")

    # default
    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadatas,
    )

    print(f"Done. Collection '{COLLECTION_NAME}' persisted to {CHROMA_DB_DIR}")


if __name__ == "__main__":
    build_index()