"""
Fetch and normalize records from the Stampy alignment-research-dataset.
Fields: source, subsource, title, authors, text, url, date_published

Running this script
python3 ingestion/fetch.py: resumes from checkpoint, skips completed sources
python3 ingestion/fetch.py --reset: clear collection and checkpoint, starts fresh
python3 ingestion/fetch.py --source lesswrong: run a single source only
"""

import argparse
import hashlib
import os
import uuid
import json
from typing import Generator

import torch  # type: ignore

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ONNX_PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]


from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# ── Chunking ──────────────────────────────────────────────────────────────────

CHUNK_SIZE = 512    # tokens
CHUNK_OVERLAP = 15  # tokens


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size chunks using whitespace tokenization."""
    tokens = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(" ".join(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


def chunk_record(record: dict) -> list[dict]:
    """Return one dict per chunk, carrying all record metadata forward."""
    chunks = chunk_text(record["text"])
    return [{**record, "text": chunk} for chunk in chunks]

# ── Embedding ─────────────────────────────────────────────────────────────────

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BM25_MODEL = "Qdrant/bm25"
VECTOR_DIM = 384
EMBED_BATCH_SIZE = 64


def build_embed_text(record: dict) -> str:
    """Construct the string to embed: title prepended to chunk text for vector embedding."""
    return f"{record['title']}\n\n{record['text']}"


def embed_records(records: list[dict], model: SentenceTransformer) -> list:
    """Return one L2-normalized dense embedding (numpy array) per record."""
    texts = [build_embed_text(r) for r in records]
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    return list(embeddings)


def embed_sparse_records(records: list[dict], model: SparseTextEmbedding) -> list[SparseVector]:
    """Return one SparseVector per record."""
    texts = [build_embed_text(r) for r in records]
    return [
        SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
        for r in model.embed(texts, batch_size=EMBED_BATCH_SIZE)
    ]


# ── Qdrant points ─────────────────────────────────────────────────────────────

def _stable_uuid(text: str) -> str:
    """Deterministic UUID from SHA-256 of text for deduplication across runs."""
    h = hashlib.sha256(text.encode()).hexdigest()
    return str(uuid.UUID(h[:32]))


def build_points(
    records: list[dict],
    dense_vecs: list,
    sparse_vecs: list[SparseVector],
) -> list[PointStruct]:
    """Combine metadata payloads and named embedding vectors into Qdrant PointStructs."""
    points: list[PointStruct] = []
    for record, dense, sparse in zip(records, dense_vecs, sparse_vecs):
        payload = {
            "source": record["source"],
            "subsource": record["subsource"],
            "title": record["title"],
            "authors": record["authors"],
            "date_published": record["date_published"],
            "url": record["url"],
            "text": record["text"],
        }
        embed_text = build_embed_text(record)
        points.append(PointStruct(
            id=_stable_uuid(embed_text),
            vector={"dense_vec": dense, "sparse_vec": sparse},
            payload=payload,
        ))
    return points


# ── Qdrant collection & upsert ────────────────────────────────────────────────
load_dotenv() # load env variables

COLLECTION_NAME = "ai_safety"
INGEST_BATCH_SIZE = 128  # chunks accumulated before embed + upsert
UPSERT_BATCH_SIZE = 128
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(client: QdrantClient, name: str = COLLECTION_NAME, reset: bool = False) -> None:
    if client.collection_exists(name):
        if reset:
            client.delete_collection(name)
            print(f"Deleted existing collection '{name}'.")
        else:
            print(f"Resuming into existing collection '{name}'.")
            return
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense_vec": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse_vec": SparseVectorParams(index=SparseIndexParams()),
        },
    )
    print(f"Created collection '{name}'.")


def upsert_in_batches(client: QdrantClient, points: list[PointStruct], name: str = COLLECTION_NAME) -> None:
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i : i + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=name, points=batch, wait=False)
        print(f"  Upserted points {i}–{i + len(batch) - 1}")


# ── Checkpoint ────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ingestion_checkpoint.json")


def load_checkpoint() -> set[str]:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return set(json.load(f).get("completed_sources", []))
    return set()


def save_checkpoint(completed: set[str]) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed_sources": sorted(completed)}, f, indent=2)


# ── Stampy data source ────────────────────────────────────────────────────────
STAMPY_SOURCES = ["lesswrong", "alignmentforum"]  # add "arxiv", "eaforum", etc.


def load_stampy(sources: list[str]) -> Generator[dict, None, None]:
    """Yield normalized records from the Stampy alignment-research-dataset JSONL files."""
    for source_name in sources:
        path = hf_hub_download(
            repo_id="StampyAI/alignment-research-dataset",
            filename=f"{source_name}.jsonl",
            repo_type="dataset",
        )
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield {
                    "source": "stampy",
                    "subsource": row.get("source", "") or source_name,
                    "title": row.get("title", "") or "",
                    "authors": row.get("authors", []) or [],
                    "text": row.get("text", "") or "",
                    "url": row.get("url", "") or "",
                    "date_published": row.get("date_published", "") or "",
                    "_source_name": source_name,
                }

# ── Ingestion entry point ─────────────────────────────────────────────────────

def _flush(
    batch: list[dict],
    dense_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    client: QdrantClient,
) -> int:
    dense_vecs = embed_records(batch, dense_model)
    print(f"  Sparse embedding {len(batch)} chunks...")
    sparse_vecs = embed_sparse_records(batch, sparse_model)
    print(f"  Sparse embedding done.")
    print("  Building points...")
    points = build_points(batch, dense_vecs, sparse_vecs)
    del dense_vecs, sparse_vecs
    print("  Upserting...")
    upsert_in_batches(client, points)
    print("  Clearing MPS cache...")
    if DEVICE == "mps":
        torch.mps.empty_cache()
    print("  Flush done.")
    return len(points)


def ingest(sources: list[str] | None = None, reset: bool = False) -> None:
    sources = sources or STAMPY_SOURCES

    completed = set() if reset else load_checkpoint()
    pending = [s for s in sources if s not in completed]
    if not pending:
        print("All sources already ingested. Use --reset to start over.")
        return
    skipped = sorted(completed & set(sources))
    if skipped:
        print(f"Skipping already-completed sources: {skipped}")
    print(f"Ingesting: {pending}")

    dense_model = SentenceTransformer(BGE_MODEL, device=DEVICE)
    if DEVICE == "mps":
        dense_model.half()
    sparse_model = SparseTextEmbedding(model_name=BM25_MODEL, providers=ONNX_PROVIDERS)
    client = get_client()
    ensure_collection(client, reset=reset)

    batch: list[dict] = []
    total = 0
    current_source: str | None = None

    for record in load_stampy(pending):
        if not record["text"].strip():
            continue

        record_source = record.pop("_source_name")

        # when source changes, flush remaining batch and checkpoint the completed source
        if current_source is not None and record_source != current_source:
            if batch:
                total += _flush(batch, dense_model, sparse_model, client)
                batch = []
            completed.add(current_source)
            save_checkpoint(completed)
            print(f"Checkpointed: '{current_source}' complete.")

        current_source = record_source
        batch.extend(chunk_record(record))
        if len(batch) >= INGEST_BATCH_SIZE:
            total += _flush(batch, dense_model, sparse_model, client)
            batch = []

    if batch:
        total += _flush(batch, dense_model, sparse_model, client)

    if current_source:
        completed.add(current_source)
        save_checkpoint(completed)
        print(f"Checkpointed: '{current_source}' complete.")

    print(f"Ingestion complete. {total} points upserted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete collection and checkpoint, start fresh.")
    parser.add_argument("--source", help="Ingest a single source (e.g. lesswrong).")
    args = parser.parse_args()

    ingest(sources=[args.source] if args.source else None, reset=args.reset)
