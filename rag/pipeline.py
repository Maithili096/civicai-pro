from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/tmp/civicai_faiss_index")

_embeddings_instance = None

def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        print("[RAG] Loading embedding model (first call)...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=_EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[RAG] Embedding model ready.")
    return _embeddings_instance


def build_index(texts: list[str], metadatas: Optional[list[dict]] = None) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "।", ".", " "],
    )
    docs: list[Document] = []
    for i, text in enumerate(texts):
        chunks = splitter.create_documents(
            [text],
            metadatas=[metadatas[i]] if metadatas else [{"source": f"doc_{i}"}],
        )
        docs.extend(chunks)

    store = FAISS.from_documents(docs, _get_embeddings())
    store.save_local(_INDEX_PATH)
    print(f"[RAG] Built index: {len(docs)} chunks → {_INDEX_PATH}")
    return store


def load_index() -> Optional[FAISS]:
    if Path(_INDEX_PATH).exists():
        return FAISS.load_local(
            _INDEX_PATH,
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return None


def retrieve(query: str, k: int = 4) -> tuple[list[str], list[str]]:
    store = load_index()
    if store is None:
        return [], []
    results = store.similarity_search_with_score(query, k=k)
    chunks  = [doc.page_content for doc, _ in results]
    sources = [doc.metadata.get("source", "unknown") for doc, _ in results]
    return chunks, sources
