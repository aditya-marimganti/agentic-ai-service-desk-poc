# Solution Design — Knowledge Agent

## 1. Overview

The Knowledge agent answers employee IT support questions (such as password policy, VPN setup, software request process, access requests, common troubleshooting) by retrieving relevant passages from an internal knowledge base and generating a grounded answer with citations. If relevant information does not exist in the knowledge base, the agent refuses to guess and instead tells the user it doesn't have that information.

## 2. Architecture / Flow

User question
   -> embed question 
   -> search ChromaDB for the most relevant chunks from knowledge base
   -> check similarity threshold
        -> if below threshold: return refusal response
        -> if above threshold: continue
   -> construct prompt: question + retrieved chunks + citation instructions
   -> send to Claude via provider abstraction (src/llm_provider.py)
   -> Claude generates answer referencing source document(s)
   -> return {answer, sources, confidence} to caller


The LLM call is routed through the provider abstraction layer so the model backend is swappable without touching agent logic, per the project brief.

## 3. Knowledge Base Design

Example documents:
1. Password reset policy
2. VPN setup and troubleshooting guide
3. Software request process
4. Access/permissions request policy
5. New hire IT onboarding checklist
6. General troubleshooting FAQ (printer, email, wifi)

Each document is a short markdown/text file, 1–3 paragraphs, stored under `data/knowledge_base/`.

**Chunking strategy:**
- Chunk by paragraph (natural semantic boundary for short policy docs)
- Fallback: fixed-size chunking (~300 tokens with ~50 token overlap) if any document proves too dense for paragraph-level chunks
- Each chunk retains metadata: `source_document`, `chunk_id`

**Embedding model:**
- ChromaDB's default embedding function to start (sentence-transformers based, runs locally, no extra API cost)
- Will revisit if retrieval quality on the eval set is poor

**Storage:**
- Local ChromaDB persistent client, stored under `data/chroma_db/`
  (excluded from git via `.gitignore`, regenerable from source docs)

## 4. Citation Approach

The agent returns a structured response, not just free text:

{
  "answer": "...",
  "sources": ["vpn_setup_guide.md", "password_reset_policy.md"],
  "confidence": "high" | "low"
}

The system prompt instructs Claude to only state facts present in the retrieved chunks and to list which source document(s) each part of the answer came from. This keeps citations grounded rather than hallucinated after the fact.

## 5. Refusal Behavior

- Compute similarity score between the question embedding and the best-matching chunk
- If the top score falls below a defined threshold (starting point: tune empirically once retrieval is running, initial guess ~0.3–0.4 cosine distance depending on ChromaDB's default metric), the agent does not call the LLM with a "best guess" — it returns a fixed refusal response: *"I don't have information on this in the knowledge base. Please contact the IT helpdesk directly."*
- This threshold is a tunable constant, expected to be adjusted after running the eval set

## 6. Evaluation Plan

Seed evaluation set: 8 scenarios, saved under `eval/seed_eval.json`

- 5 questions with a clear correct answer in the knowledge base (tests retrieval + grounded generation + citation accuracy)
- 3 questions with no relevant answer in the knowledge base (tests refusal behavior — the agent must not hallucinate)

**Target:** ≥85% grounded accuracy across the 8 scenarios (correct answer AND correct citation AND correct refusal where applicable).

## 7. Provider Abstraction

LLM calls are made through `src/llm_provider.py`, a single `generate()` function wrapping the Anthropic API. This satisfies the brief's requirement that the LLM be swappable behind an abstraction — a different provider can be substituted by editing this one file, with no changes needed in agent logic elsewhere.

## 8. Open Questions / Risks

- Exact similarity threshold for refusal is unset — needs empirical tuning once the eval set is running against real retrieval results
- Chunking strategy (paragraph vs fixed-size) may need revisiting depending on how the mock documents are actually structured once written
- Not yet decided whether citations should be per-sentence or per-answer — starting with per-answer (simpler) and revisiting if eval results show citation accuracy issues