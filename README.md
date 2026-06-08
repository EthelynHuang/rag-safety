---
title: AI Safety RAG
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI Safety RAG

A retrieval-augmented generation (RAG) system for AI safety and alignment research. Ask any question about AI alignment — the system retrieves the most relevant passages from **5,823 papers and posts** across LessWrong and the Alignment Forum, then generates a grounded, cited answer using Claude.

## What It Does

Type a question like *"What is Goodhart's Law and why does it matter for AI alignment?"* and the system:

1. Retrieves the top candidate passages from a 28,000-chunk vector database using hybrid dense + sparse search
2. Reranks them with a cross-encoder to surface the most relevant context
3. Generates a concise answer using Claude, with inline citations tied to source documents

Every claim in the answer links back to the original paper or post.

## Architecture

```
Query
  │
  ▼
Hybrid Retrieval (Qdrant)
  ├─ Dense vectors  — BAAI/bge-small-en-v1.5 (cosine similarity)
  └─ Sparse vectors — BM25 (term matching)
        │
        └─ Reciprocal Rank Fusion (RRF) → top 10 candidates
              │
              ▼
        Cross-Encoder Reranker
        cross-encoder/ms-marco-MiniLM-L-6-v2 → top 5 passages
              │
              ▼
        Answer Generation
        Claude (claude-sonnet-4-6) with grounded system prompt
              │
              ▼
        Answer + cited Sources
```

**Key design decisions** 
- I chose a hybrid search with dense retrieval for semantic similarity, and BM25 for precise keyword matches. I fused both methods with RRF to identify relevant papers with named concepts and context considered
- I also incorporated a cross-encoder reranker in the initial chunk fetch to attend over both the query and document simultaneously, providing a stronger relevance signal over a bi-encoder retrieval

## Evaluation

Evaluated with [RAGAS](https://github.com/explodinggradients/ragas) on a curated set of AI safety questions, judged by Claude Haiku acting as an LLM evaluator.

| Pipeline | Faithfulness | Context Precision |
|---|---|---|
| Dense-only | 0.970 | 0.675 |
| **Hybrid + Reranker** | **0.966** | **0.758** |

- **Faithfulness** — fraction of answer claims that are grounded in the retrieved context (higher = less hallucination)
- **Context Precision** — fraction of retrieved chunks that were actually relevant to the question (higher = less noise)

Hybrid + reranker achieves a **+8.3 pp improvement in context precision** over dense-only retrieval, meaning the model receives cleaner context and produces more focused answers.

## Data

- **Source:** [StampyAI/alignment-research-dataset](https://huggingface.co/datasets/StampyAI/alignment-research-dataset)
- **Corpora:** LessWrong, Alignment Forum
- **Documents:** 5,823 unique papers and posts
- **Chunks:** 28,513 (512-token windows, 15-token overlap)
- **Vector store:** Qdrant Cloud (named vectors, separate dense + sparse indices per chunk)

Ingestion is resumable via checkpointing; documents are deduplicated by SHA-256 of their embed text, so re-runs are idempotent.

## Stack

| Layer | Technology |
|---|---|
| Vector store | Qdrant Cloud |
| Dense embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| Sparse embeddings | BM25 (fastembed) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Anthropic Claude (claude-sonnet-4-6) |
| API | FastAPI + Pydantic |
| UI | Gradio (mounted on FastAPI) |
| Deployment | Docker → HuggingFace Spaces |
| Evaluation | RAGAS + LangChain |

## API

```
POST /query
Content-Type: application/json

{ "query": "What is mesa-optimization?" }
```

```json
{
  "answer": "Mesa-optimization refers to... [1][2]",
  "sources": [
    { "title": "Risks from Learned Optimization", "url": "...", "authors": [...] },
    ...
  ]
}
```

## Running Locally

```bash
# Set environment variables
export QDRANT_URL=<your-qdrant-url>
export QDRANT_API_KEY=<your-api-key>
export ANTHROPIC_API_KEY=<your-anthropic-key>

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn serving.api:app --host 0.0.0.0 --port 7860
```

To re-ingest the corpus:

```bash
python3 ingestion/fetch.py           # resume from checkpoint
python3 ingestion/fetch.py --reset   # start fresh
python3 ingestion/fetch.py --source lesswrong  # single source
```

To run the eval suite:

```bash
python3 eval/run_eval.py              # hybrid + reranker
python3 eval/run_eval.py --dense-only # ablation: dense only
```
