
import sys
import json
import chromadb
from llm_provider import generate

CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "it_knowledge_base"

MAX_DISTANCE_THRESHOLD = 1.3

TOP_K = 3  # how many chunks to retrieve per question

SYSTEM_PROMPT = """You are the Knowledge agent for an internal IT service desk.

You will be given a user's question along with some retrieved excerpts
from the company's internal IT documentation.

Rules:
- Only answer using information present in the provided excerpts.
- Do not use outside knowledge or make assumptions beyond what's stated.
- For every part of your answer, note which source document it came from.
- If the excerpts don't actually contain enough information to answer
  the question, say so honestly rather than guessing.

Respond ONLY with valid JSON in this exact format, no other text:
{
  "answer": "your answer here",
  "sources": ["filename1.md", "filename2.md"],
  "confidence": "high" or "low"
}
"""


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve(question: str, collection, top_k: int = TOP_K) -> dict:
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )
    return results


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences if Claude wraps its JSON in them."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return cleaned


def ask(question: str) -> dict:
    collection = get_collection()
    results = retrieve(question, collection)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents or distances[0] > MAX_DISTANCE_THRESHOLD:
        return {
            "answer": "I don't have information on this in the knowledge base. Please contact the IT helpdesk directly.",
            "sources": [],
            "confidence": "low",
            "_debug_top_distance": distances[0] if distances else None,
        }

    context_parts = []
    for doc_text, meta in zip(documents, metadatas):
        context_parts.append(f"[Source: {meta['source_document']}]\n{doc_text}")
    context = "\n\n".join(context_parts)

    user_message = f"""Question: {question}

Retrieved excerpts:
{context}
"""

    raw_response = generate(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    cleaned = strip_code_fences(raw_response)

    try:
        parsed = json.loads(cleaned)
        parsed["_debug_top_distance"] = distances[0]
        return parsed
    except json.JSONDecodeError:
        # Fallback 
        return {
            "answer": raw_response,
            "sources": [m["source_document"] for m in metadatas],
            "confidence": "low",
            "_debug_note": "Failed to parse structured JSON from model response",
            "_debug_top_distance": distances[0],
        }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask a question: ")

    result = ask(question)
    print(json.dumps(result, indent=2))