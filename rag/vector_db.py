"""
rag/vector_db.py
─────────────────
Persistent ChromaDB vector database.
Replaces in-memory FAISS with a persistent, queryable vector store.

Install: pip install chromadb
"""
from __future__ import annotations
from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_CHROMA_DIR  = "civicai_chroma_db"
_COLLECTION  = "civicai_docs"

def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectordb() -> Chroma:
    """Returns persistent Chroma vector store (creates if not exists)."""
    return Chroma(
        collection_name=_COLLECTION,
        embedding_function=_get_embeddings(),
        persist_directory=_CHROMA_DIR,
    )

def ingest(texts: list[str], metadatas: Optional[list[dict]] = None) -> int:
    """Add documents to ChromaDB. Returns total doc count."""
    db = get_vectordb()
    metas = metadatas or [{"source": f"doc_{i}"} for i in range(len(texts))]
    db.add_texts(texts=texts, metadatas=metas)
    return db._collection.count()

def retrieve(query: str, k: int = 4) -> tuple[list[str], list[str]]:
    """Returns (chunks, sources) for top-k similar docs."""
    db = get_vectordb()
    if db._collection.count() == 0:
        return [], []
    results = db.similarity_search_with_score(query, k=k)
    chunks  = [doc.page_content for doc, _ in results]
    sources = [doc.metadata.get("source", "unknown") for doc, _ in results]
    return chunks, sources

def list_docs() -> list[dict]:
    """Lists all documents in the vector store."""
    db = get_vectordb()
    data = db._collection.get(include=["metadatas", "documents"])
    return [
        {"id": i, "source": m.get("source", ""), "preview": d[:100]}
        for i, (m, d) in enumerate(zip(data["metadatas"], data["documents"]))
    ]

def delete_collection() -> None:
    """Wipes the entire collection."""
    db = get_vectordb()
    db.delete_collection()
