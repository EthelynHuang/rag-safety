"""
Fetch and normalize records from the Stampy alignment-research-dataset.
Fields: source, subsource, title, authors, text, url, date_published
"""

import hashlib
import os
import uuid
from typing import Generator

from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
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

BGE_MODEL = "BAAI/bge-base-en-v1.5"
VECTOR_DIM = 768
EMBED_BATCH_SIZE = 64


def build_embed_text(record: dict) -> str:
    """Construct the string to embed: title prepended to chunk text for vector embedding."""
    return f"{record['title']}\n\n{record['text']}"


def embed_records(records: list[dict], model: SentenceTransformer) -> list[list[float]]:
    """Return one normalized embedding vector per record."""
    texts = [build_embed_text(r) for r in records]
    return model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()


# ── Qdrant points ─────────────────────────────────────────────────────────────

def _stable_uuid(text: str) -> str:
    """Deterministic UUID from SHA-256 of text for deduplication across runs."""
    h = hashlib.sha256(text.encode()).hexdigest()
    return str(uuid.UUID(h[:32]))


def build_points(records: list[dict], vectors: list[list[float]]) -> list[PointStruct]:
    """Combine metadata payloads and embedding vectors into Qdrant PointStructs."""
    points: list[PointStruct] = []
    for record, vector in zip(records, vectors):
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
        points.append(PointStruct(id=_stable_uuid(embed_text), vector=vector, payload=payload))
    return points


# ── Qdrant collection & upsert ────────────────────────────────────────────────

COLLECTION_NAME = "ai_safety"
INGEST_BATCH_SIZE = 512  # chunks accumulated before embed + upsert
UPSERT_BATCH_SIZE = 64
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(client: QdrantClient, name: str = COLLECTION_NAME) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection '{name}'.")
    else:
        print(f"Collection '{name}' already exists.")


def upsert_in_batches(client: QdrantClient, points: list[PointStruct], name: str = COLLECTION_NAME) -> None:
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i : i + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=name, points=batch)
        print(f"  Upserted points {i}–{i + len(batch) - 1}")


# ── Stampy data source ────────────────────────────────────────────────────────

def load_stampy() -> Generator[dict, None, None]:
    """Yield normalized records from the Stampy alignment-research-dataset."""
    dataset = load_dataset("StampyAI/alignment-research-dataset", trust_remote_code=True)
    splits = list(dataset.values()) if hasattr(dataset, "values") else [dataset]
    for split in splits:
        for row in split:
            yield {
                "source": "stampy",
                "subsource": row.get("source", ""),
                "title": row.get("title", "") or "",
                "authors": row.get("authors", []) or [],
                "text": row.get("text", "") or "",
                "url": row.get("url", "") or "",
                "date_published": row.get("date_published", "") or "",
            }


# ── Ingestion entry point ─────────────────────────────────────────────────────

def _flush(batch: list[dict], model: SentenceTransformer, client: QdrantClient) -> int:
    vectors = embed_records(batch, model)
    points = build_points(batch, vectors)
    upsert_in_batches(client, points)
    return len(points)


def ingest() -> None:
    model = SentenceTransformer(BGE_MODEL)
    client = get_client()
    ensure_collection(client)

    print("Ingesting Stampy dataset...")
    batch: list[dict] = []
    total = 0

    for record in load_stampy():
        if not record["text"].strip():
            continue
        batch.extend(chunk_record(record))
        if len(batch) >= INGEST_BATCH_SIZE:
            total += _flush(batch, model, client)
            batch = []

    if batch:
        total += _flush(batch, model, client)

    print(f"Ingestion complete. {total} points upserted.")


if __name__ == "__main__":
    ingest()
